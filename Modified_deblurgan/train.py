import time
import easydict
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from multiprocessing import freeze_support
import os

def print_current_errors(epoch, i, errors, t, opt):
	message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
	for k, v in errors.items():
		message += '%s: %.3f ' % (k, v)

	print(message)
	log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
	with open(log_name, "a") as log_file:
		log_file.write('%s\n' % message)

def train(opt, data_loader, model, visualizer):
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	total_steps = 0
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Restored_Train'], results['Sharp_Train'])
				print('PSNR on Train = %f, Learning_rate = %.4f' % (psnrMetric, model.old_lr))
				# visualizer.display_current_results(results, epoch)

			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				t = (time.time() - iter_start_time) / opt.batchSize
				print_current_errors(epoch, epoch_iter, errors, t, opt)

			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				model.save('latest')

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save('latest')
			model.save(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch == 50:
			model.update_learning_rate()



if __name__ == '__main__':
	freeze_support()

	opt = easydict.EasyDict({
		#train option
		"lr" : 0.0005,
		"save_latest_freq" : 1000,
		"save_epoch_freq" : 1,
		"continue_train" : False,
		"epoch_count" : 1,
		"phase" : "train",
		"which_epoch" : "latest",
		"niter" : 70,
		"niter_decay" : 30,
		"beta1" : 0.5,
		"lambda_A" : 100.0,
		"lambda_B" : 10.0,
		"identity" : 0.0,
		"pool_size" : 50,
		"no_html" : None,

		#base option
		"dataroot" : "train_no_flipping/",
		"learn_residual" : True,
		"dataset_mode" : "unaligned",
		"checkpoints_dir" : "checkpoints/",
		"batchSize": 4,
		"loadSizeX": 256,
		"loadSizeY": 256,
		"fineSize": 256,
		"input_nc": 3,
		"output_nc": 3,
		"ngf" : 64,
		"ndf" : 64,
		"which_model_netD" : "basic",
		"which_model_netG" : "resnet_6blocks",
		"gan_type" : "wgan-gp",
		"n_layers_D" : 3,
		"gpu_ids" : [0],
		"name" : "",
		"model" : "content_gan",
		"which_direction" : "AtoB",
		"nThreads" : 4,
		"norm" : "batch",
		"serial_batches" : True,
		"isTrain" : True,
		"no_dropout" : True,
		"max_dataset_size" : float("inf"),
		"display_freq": 100,
		"print_freq": 100,
		"display_id" : 1,
		"display_port" : 8097,
		"display_winsize" : 256,
		"display_single_pane_ncols" : 0,
		"resize_or_crop" : "resize"


	})
	opt.learn_residual = True
	opt.fineSize = 256
	opt.gan_type ="wgan-gp"

	data_loader = CreateDataLoader(opt)
	model = create_model(opt)
	visualizer = Visualizer(opt)
	train(opt, data_loader, model, Visualizer)

