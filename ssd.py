# coding=utf-8
import tensorflow as tf
import numpy as np
from loss import MultiboxLoss
from ssd_utils import BBoxUtility, Generator, preprocess_input
import h5py, pickle, os, cv2, config, sys

activation = tf.nn.relu
FLAGS = tf.app.flags.FLAGS

class SSD:
	def __init__(self, input_shape = (300, 300, 3)):
		self.num_class = config.NUM_CLASSES
		self.input_tensor = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]])
		self.label_tensor = tf.placeholder(tf.float32, [None, 7308, 4 + config.NUM_CLASSES + 8])
		self.predicts = self.build(input_shape, config.NUM_CLASSES)
		self.input_shape = input_shape
		self.global_step = tf.train.create_global_step()
		var_list = tf.global_variables()
		var_list = [var for var in var_list if "Adam" not in var.name]
		self.saver = tf.train.Saver(var_list, max_to_keep=1)
		self.bbox_util = BBoxUtility(self.num_class)

	def build(self, input_shape, num_classes):
		img_size = (input_shape[1], input_shape[0])
		#300
		conv1_1 = tf.layers.conv2d(self.input_tensor, 64, 3, name = "conv1_1", padding = "same", activation = activation)
		self.conv1_1 = conv1_1
		conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, name = "conv1_2", padding = "same", activation = activation)
		pool1 = tf.layers.max_pooling2d(conv1_2, pool_size = 2, strides = 2, padding = "same")
		#150
		conv2_1 = tf.layers.conv2d(pool1, 128, 3, name = "conv2_1", padding = "same", activation = activation)
		conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, name = "conv2_2", padding = "same", activation = activation)
		pool2 = tf.layers.max_pooling2d(conv2_2, pool_size = 2, strides = 2, padding = "same")
		#75
		conv3_1 = tf.layers.conv2d(pool2, 256, 3, name = "conv3_1", padding = "same", activation = activation)
		conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, name = "conv3_2", padding = "same", activation = activation)
		conv3_3 = tf.layers.conv2d(conv3_2, 256, 3, name = "conv3_3", padding = "same", activation = activation)
		pool3 = tf.layers.max_pooling2d(conv3_3, pool_size = 2, strides = 2, padding = "same")
		#38
		conv4_1 = tf.layers.conv2d(pool3, 512, 3, name = "conv4_1", padding = "same", activation = activation)
		conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, name = "conv4_2", padding = "same", activation = activation)
		conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, name = "conv4_3", padding = "same", activation = activation)
		pool4 = tf.layers.max_pooling2d(conv4_3, pool_size = 2, strides = 2, padding = "same")
		#19
		conv5_1 = tf.layers.conv2d(pool4, 512, 3, name = "conv5_1", padding = "same", activation = activation)
		conv5_2 = tf.layers.conv2d(conv5_1, 512, 3, name = "conv5_2", padding = "same", activation = activation)
		conv5_3 = tf.layers.conv2d(conv5_2, 512, 3, name = "conv5_3", padding = "same", activation = activation)
		pool5 = tf.layers.max_pooling2d(conv5_3, pool_size = 3, strides = 1, padding = "same")
		#19
		fc6_kernel = tf.get_variable(name = "fc6/kernel", shape = (3, 3, 512, 1024), initializer = tf.truncated_normal_initializer(stddev=0.1))
		fc6_bias = tf.get_variable(name = "fc6/bias", shape = [1024], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		fc6 = tf.nn.atrous_conv2d(pool5, fc6_kernel, rate = 6, padding = "SAME", name = "fc6")
		fc6 = tf.nn.bias_add(fc6, fc6_bias)
		fc6 = activation(fc6)

		fc7 = tf.layers.conv2d(fc6, 1024, 1, name = "fc7", padding = "same", activation = activation)

		conv6_1 = tf.layers.conv2d(fc7, 256, 1, name = "conv6_1", padding = "same", activation = activation)
		conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, name = "conv6_2", strides = (2,2), padding = "same", activation = activation)
		#10
		conv7_1 = tf.layers.conv2d(conv6_2, 128, 1, name = "conv7_1", padding = "same", activation = activation)
		conv7_2 = tf.keras.layers.ZeroPadding2D()(conv7_1)
		conv7_2 = tf.layers.conv2d(conv7_2, 256, 3, name = "conv7_2", padding = "valid", strides = (2,2), activation = activation)
		#5
		conv8_1 = tf.layers.conv2d(conv7_2, 128, 1, name = "conv8_1", padding = "same", activation = activation)
		conv8_2 = tf.layers.conv2d(conv8_1, 256, 3, name = "conv8_2", padding = "same", strides = (2,2), activation = activation)
		#3
		pool6 = tf.keras.layers.GlobalAveragePooling2D(name='pool6')(conv8_2)
		#1
		num_priors = 3
		conv4_3_norm = self.normalize_layer(conv4_3, 20, 512, "conv4_3_norm")
		conv4_3_norm_mbox_loc = tf.layers.conv2d(conv4_3_norm, num_priors * 4, 3, name = "conv4_3_norm_mbox_loc", padding = "same")
		conv4_3_norm_mbox_loc_flat = tf.layers.flatten(conv4_3_norm_mbox_loc)
		name = "conv4_3_norm_mbox_conf"
		if num_classes!=21:
			name+="_"+str(num_classes)
		conv4_3_norm_mbox_conf = tf.layers.conv2d(conv4_3_norm, num_priors * num_classes, 3, name = name, padding = "same")
		conv4_3_norm_mbox_conf_flat = tf.layers.flatten(conv4_3_norm_mbox_conf)
		shape = [0, 38, 38, 512]
		conv4_3_norm_mbox_priorbox = self.priorBox_layer(conv4_3_norm, shape, img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')

		num_priors = 6
		fc7_mbox_loc = tf.layers.conv2d(fc7, num_priors * 4, 3, name = "fc7_mbox_loc", padding = "same")
		fc7_mbox_loc_flat = tf.layers.flatten(fc7_mbox_loc)
		name = "fc7_mbox_conf"
		if num_classes!=21:
			name+="_"+str(num_classes)	
		fc7_mbox_conf = tf.layers.conv2d(fc7, num_priors * num_classes, 3, name = name, padding = "same")
		fc7_mbox_conf_flat = tf.layers.flatten(fc7_mbox_conf)
		shape = [0, 19, 19, 1024]
		fc7_mbox_priorbox = self.priorBox_layer(fc7, shape, img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')

		num_priors = 6
		conv6_2_mbox_loc = tf.layers.conv2d(conv6_2, num_priors * 4, 3, name = "conv6_2_mbox_loc", padding = "same")
		conv6_2_mbox_loc_flat = tf.layers.flatten(conv6_2_mbox_loc)
		name = "conv6_2_mbox_conf"
		if num_classes!=21:
			name+="_"+str(num_classes)	
		conv6_2_mbox_conf = tf.layers.conv2d(conv6_2, num_priors * num_classes, 3, name = name, padding = "same")
		conv6_2_mbox_conf_flat = tf.layers.flatten(conv6_2_mbox_conf)
		shape = [0, 10, 10, 256]
		conv6_2_mbox_priorbox = self.priorBox_layer(conv6_2, shape, img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')

		num_priors = 6
		conv7_2_mbox_loc = tf.layers.conv2d(conv7_2, num_priors * 4, 3, name = "conv7_2_mbox_loc", padding = "same")
		conv7_2_mbox_loc_flat = tf.layers.flatten(conv7_2_mbox_loc)
		name = "conv7_2_mbox_conf"
		if num_classes!=21:
			name+="_"+str(num_classes)		
		conv7_2_mbox_conf = tf.layers.conv2d(conv7_2, num_priors * num_classes, 3, name = name, padding = "same")
		conv7_2_mbox_conf_flat = tf.layers.flatten(conv7_2_mbox_conf)
		shape = [0, 5, 5, 256]
		conv7_2_mbox_priorbox = self.priorBox_layer(conv7_2, shape, img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')

		num_priors = 6
		conv8_2_mbox_loc = tf.layers.conv2d(conv8_2, num_priors * 4, 3, name = "conv8_2_mbox_loc", padding = "same")
		conv8_2_mbox_loc_flat = tf.layers.flatten(conv8_2_mbox_loc)
		name = "conv8_2_mbox_conf"
		if num_classes!=21:
			name+="_"+str(num_classes)		
		conv8_2_mbox_conf = tf.layers.conv2d(conv8_2, num_priors * num_classes, 3, name = name, padding = "same")
		conv8_2_mbox_conf_flat = tf.layers.flatten(conv8_2_mbox_conf)
		shape = [0, 3, 3, 256]
		conv8_2_mbox_priorbox = self.priorBox_layer(conv8_2, shape, img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')

		num_priors = 6
		pool6_mbox_loc_flat = tf.layers.dense(pool6, units = num_priors * 4, name='pool6_mbox_loc_flat')
		name = "pool6_mbox_conf_flat"
		if num_classes!=21:
			name+="_"+str(num_classes)	
		pool6_mbox_conf_flat = tf.layers.dense(pool6, units = num_priors * num_classes, name=name)
		shape = [0, 1, 1, 256]
		pool6_mbox_priorbox = self.priorBox_layer(tf.reshape(pool6, (-1, 1, 1, 256)), shape, img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')

		mbox_loc = tf.concat([conv4_3_norm_mbox_loc_flat, 
							fc7_mbox_loc_flat, 
							conv6_2_mbox_loc_flat,
							conv7_2_mbox_loc_flat,
							conv8_2_mbox_loc_flat,
							pool6_mbox_loc_flat], axis = 1)

		mbox_conf = tf.concat([conv4_3_norm_mbox_conf_flat,
							fc7_mbox_conf_flat,
							conv6_2_mbox_conf_flat,
							conv7_2_mbox_conf_flat,
							conv8_2_mbox_conf_flat,
							pool6_mbox_conf_flat], axis = 1)

		mbox_priorbox = tf.concat([conv4_3_norm_mbox_priorbox,
								fc7_mbox_priorbox,
								conv6_2_mbox_priorbox,
								conv7_2_mbox_priorbox,
								conv8_2_mbox_priorbox,
								pool6_mbox_priorbox], axis=1)
		mbox_priorbox = tf.cast(mbox_priorbox, tf.float32)

		num_boxes = tf.shape(mbox_loc)[-1]//4
		mbox_loc = tf.reshape(mbox_loc, (-1, num_boxes, 4))
		mbox_conf = tf.reshape(mbox_conf, (-1, num_boxes, num_classes))
		mbox_conf = tf.nn.softmax(mbox_conf)
		predictions = tf.concat([mbox_loc, mbox_conf, mbox_priorbox], axis = 2)

		return predictions

	def normalize_layer(self, net, init_scale, shape=512, name = None):
		init_scale = init_scale * np.ones(shape)
		scale = tf.Variable(init_scale, name = name, dtype = tf.float32)
		return scale * tf.nn.l2_normalize(net, 3)

	def priorBox_layer(self, net, input_shape, img_size, min_size, max_size=None, aspect_ratios=None, flip=True, variances=[0.1], clip=True, name = None):
		aspect_ratios_ = [1.0]
		if max_size:
			if max_size < min_size:
				raise Exception('max_size must be greater than min_size.')
			aspect_ratios_.append(1.0)
		if aspect_ratios:
			for ar in aspect_ratios:
				if ar in aspect_ratios_: continue
				aspect_ratios_.append(ar)
				if flip:
					aspect_ratios_.append(1.0 / ar)
		variances = np.array(variances)
		layer_width = input_shape[2]
		layer_height = input_shape[1]
		img_width = img_size[0]
		img_height = img_size[1]
		box_widths = []
		box_heights = []
		for ar in aspect_ratios_:
			if ar == 1 and len(box_widths) == 0:
				box_widths.append(min_size)
				box_heights.append(min_size)
			elif ar == 1 and len(box_widths) > 0:
				box_widths.append(np.sqrt(min_size * max_size))
				box_heights.append(np.sqrt(min_size * max_size))
			elif ar != 1:
				box_widths.append(min_size * np.sqrt(ar))
				box_heights.append(min_size / np.sqrt(ar))
		box_widths = 0.5 * np.array(box_widths)
		box_heights = 0.5 * np.array(box_heights)
		step_x = img_width / layer_width
		step_y = img_height / layer_height
		linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
		liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

		centers_x, centers_y = np.meshgrid(linx, liny)
		centers_x = centers_x.reshape(-1, 1)
		centers_y = centers_y.reshape(-1, 1)
		num_priors_ = len(aspect_ratios_)
		prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
		prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
		prior_boxes[:, ::4] -= box_widths
		prior_boxes[:, 1::4] -= box_heights
		prior_boxes[:, 2::4] += box_widths
		prior_boxes[:, 3::4] += box_heights
		prior_boxes[:, ::2] /= img_width
		prior_boxes[:, 1::2] /= img_height
		prior_boxes = prior_boxes.reshape(-1, 4)
		if clip:
			prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
		num_boxes = len(prior_boxes)
		if len(variances) == 1:
			variances = np.ones((num_boxes, 4)) * variances[0]
		elif len(variances) == 4:
			variances = np.tile(variances, (num_boxes, 1))
		else:
			raise Exception('Must provide one or four variances.')
		prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
		prior_boxes_tensor = tf.expand_dims(tf.Variable(prior_boxes, name = name), 0)
		pattern = [tf.shape(net)[0], 1, 1]
		prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)
		return prior_boxes_tensor

	def restore(self, sess):
		checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
		if checkpoint:
			print("restore from: " + checkpoint)
			self.saver.restore(sess, checkpoint)
		### if you don't want to use this pretrained weights and don't want to install h5py, you can comment this block ##
		elif os.path.exists('weights_SSD300.hdf5'):
			print("restore from pretrained weights")
			tf_variables = {}
			ops = []
			for variables in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
				if "Adam" in variables.name or "RMS" in variables.name: continue
				key = variables.name.split("/")[0].split(":")[0]
				if key not in tf_variables:
					tf_variables[key] = [variables]
				else:
					tf_variables[key].append(variables)
			with h5py.File('weights_SSD300.hdf5','r') as f:
				for k in f.keys():
					if k in tf_variables:
						nn = 0
						for kk in f[k].keys():
							a = np.array(f[k][kk])
							ops.append(tf_variables[k][nn].assign(a))
							nn+=1
			sess.run(ops)
		######################################## end ####################################################################
		elif os.path.exists("vgg16.npy"):
			print("restore from vgg weights.")
			vgg = np.load("vgg16.npy", encoding='latin1').item()
			ops = []
			vgg_dict = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2", "conv4_3",
			"conv5_1","conv5_2","conv5_3"]
			tf_variables = {}
			for variables in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
				if "Adam" or "RMS" in variables.name: continue
				key = variables.name.split("/")[0].split(":")[0]
				if key not in vgg_dict: continue
				if key not in tf_variables:
					tf_variables[key] = [variables]
					ops.append(variables.assign(vgg[key][0]))
				else:
					tf_variables[key].append(variables)
					ops.append(variables.assign(vgg[key][1]))
			sess.run(ops)
		else:
			print("train from scratch.")

	def train(self):
		self.loss = MultiboxLoss(self.num_class, neg_pos_ratio=2.0).compute_loss(self.label_tensor, self.predicts)
		self.loss_avg = tf.reduce_mean(self.loss)
		
		learning_rate = tf.train.exponential_decay(config.lr, self.global_step, 10000 ,0.9, True, name='learning_rate')
		self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step = self.global_step)
		self.train_loss_summary = tf.summary.scalar("loss_train", self.loss_avg)
		self.val_loss_summary = tf.summary.scalar("loss_val", self.loss_avg)
		self.writer = tf.summary.FileWriter(FLAGS.checkpoint)

		priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
		self.bbox_util = BBoxUtility(self.num_class, priors)

		gt = pickle.load(open(FLAGS.label_file, 'rb'))
		keys = sorted(gt.keys())
		num_train = int(round(0.8 * len(keys)))
		train_keys = keys[:num_train]
		val_keys = keys[num_train:]

		gen = Generator(gt, self.bbox_util, config.BATCH_SIZE, FLAGS.images_dir,
		                train_keys, val_keys,
		                (self.input_shape[0], self.input_shape[1]))#, do_crop=False, saturation_var = 0, brightness_var = 0, contrast_var = 0, lighting_std = 0, hflip_prob = 0, vflip_prob = 0)
		c = tf.ConfigProto()
		c.gpu_options.allow_growth = True
		with tf.Session(config=c) as sess:
			sess.run(tf.global_variables_initializer())
			self.writer.add_graph(sess.graph)
			self.restore(sess)
			for inputs, labels in gen.generate(True):
				_, lo, step, summary = sess.run([self.train_op, self.loss_avg, self.global_step, self.train_loss_summary], feed_dict = {self.input_tensor: inputs, self.label_tensor: labels})
				sys.stdout.write("train loss: %d %.3f \r"%(step, lo))
				sys.stdout.flush()
				self.writer.add_summary(summary, step)
				if step % config.save_step == config.save_step - 1:
					self.saver.save(sess, os.path.join(FLAGS.checkpoint, "ckpt"), global_step=self.global_step)
					print("saved")
				if step % config.snapshot_step == 0:
					val_in, val_la = next(gen.generate(False))
					lo, s, preds = sess.run([self.loss_avg, self.train_loss_summary, self.predicts], feed_dict = {self.input_tensor: val_in, self.label_tensor: val_la})
					self.writer.add_summary(s, step)
					print("val loss:", step, lo)
					images = [np.array(val_in[v]) for v in range(val_in.shape[0])]
					self.paint_imgs(preds, images)

		print("Train finished. Checkpoint saved in", FLAGS.checkpoint)

	def predict(self):
		inputs = []
		images = []
		file_name = []
		file_list = os.listdir(FLAGS.images_dir)
		for file in file_list:
			img_path = os.path.join(FLAGS.images_dir, file)
			img = cv2.imread(img_path)
			images.append(img.copy())
			img = cv2.resize(img, (300, 300)).astype(np.float32)
			inputs.append(img)
			file_name.append(file)
		inputs = np.array(inputs)
		inputs = preprocess_input(np.array(inputs))

		c = tf.ConfigProto()
		c.gpu_options.allow_growth = True
		with tf.Session(config=c) as sess:
			init = tf.global_variables_initializer()
			sess.run(init)
			self.restore(sess)
			#todo batch
			preds = sess.run(self.predicts, feed_dict = {self.input_tensor: inputs})
			self.paint_imgs(preds, images, file_name)
		print("Finished. Images saved in " + FLAGS.eval_output_dir)

	def paint_imgs(self, preds, images, file_name=None):
		results = self.bbox_util.detection_out(preds)
		for j, img in enumerate(images):
			# Parse the outputs.
			det_label = results[j][:, 0]
			det_conf = results[j][:, 1]
			det_xmin = results[j][:, 2]
			det_ymin = results[j][:, 3]
			det_xmax = results[j][:, 4]
			det_ymax = results[j][:, 5]

			# Get detections with confidence higher than config.visual_threshold.
			top_indices = [i for i, conf in enumerate(det_conf) if conf >= config.visual_threshold]

			top_conf = det_conf[top_indices]
			top_label_indices = det_label[top_indices].tolist()
			top_xmin = det_xmin[top_indices]
			top_ymin = det_ymin[top_indices]
			top_xmax = det_xmax[top_indices]
			top_ymax = det_ymax[top_indices]

			for i in range(top_conf.shape[0]):
				xmin = int(round(top_xmin[i] * img.shape[1]))
				ymin = int(round(top_ymin[i] * img.shape[0]))
				xmax = int(round(top_xmax[i] * img.shape[1]))
				ymax = int(round(top_ymax[i] * img.shape[0]))
				score = top_conf[i]
				label = int(top_label_indices[i])
				label_name = config.CLASS_NAMES[label - 1]
				display_txt = '{:0.2f}, {}'.format(score, label_name)
				coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
				cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
				cv2.putText(img, display_txt, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,64), 1)
			if not file_name:
				name = str(j)+".jpg"
			else:
				name = file_name[j]
			cv2.imwrite(os.path.join(FLAGS.eval_output_dir, name), img)

if __name__ == '__main__':
	model = SSD()
	if FLAGS.mode == "train":
		model.train()
	else:
		model.predict()
