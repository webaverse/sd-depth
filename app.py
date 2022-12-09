import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.util import AddMiDaS

from flask import Flask, Response, request

import cv2
import io


app = Flask(__name__)
torch.set_grad_enabled(False)


def initialize_model(config, ckpt):
	config = OmegaConf.load(config)
	model = instantiate_from_config(config.model)
	model.load_state_dict(torch.load(ckpt)['state_dict'], strict=False)

	device = torch.device(
		'cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = model.to(device)
	sampler = DDIMSampler(model)
	return sampler

def make_batch_sd(
	image,
	txt,
	device,
	num_samples=1,
	model_type='dpt_hybrid'
):
	# image = np.array(image.convert('RGB'))
	image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
	# sample['jpg'] is tensor hwc in [-1, 1] at this point
	midas_trafo = AddMiDaS(model_type=model_type)
	batch = {
		'jpg': image,
		'txt': num_samples * [txt],
	}
	batch = midas_trafo(batch)
	batch['jpg'] = rearrange(batch['jpg'], 'h w c -> 1 c h w')
	batch['jpg'] = repeat(batch['jpg'].to(device=device),
						  '1 ... -> n ...', n=num_samples)
	batch['midas_in'] = repeat(torch.from_numpy(batch['midas_in'][None, ...]).to(
		device=device), '1 ... -> n ...', n=num_samples)
	return batch

def paint(sampler, image, prompt, t_enc, seed, scale, num_samples=1, callback=None, do_full_sample=False):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = sampler.model
	seed_everything(seed)

	print('Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...')
	wm = 'SDV2'
	wm_encoder = WatermarkEncoder()
	wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

	with torch.no_grad(),\
			torch.autocast('cuda'):
		batch = make_batch_sd(
			image, txt=prompt, device=device, num_samples=num_samples)
		z = model.get_first_stage_encoding(model.encode_first_stage(
			batch[model.first_stage_key]))  # move to latent space
		c = model.cond_stage_model.encode(batch['txt'])
		c_cat = list()
		for ck in model.concat_keys:
			cc = batch[ck]
			cc = model.depth_model(cc)
			depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
																						   keepdim=True)
			display_depth = (cc - depth_min) / (depth_max - depth_min)
			depth_image = Image.fromarray(
				(display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
			cc = torch.nn.functional.interpolate(
				cc,
				size=z.shape[2:],
				mode='bicubic',
				align_corners=False,
			)
			depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
																						   keepdim=True)
			cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
			c_cat.append(cc)
		c_cat = torch.cat(c_cat, dim=1)
		# cond
		cond = {'c_concat': [c_cat], 'c_crossattn': [c]}

		# uncond cond
		uc_cross = model.get_unconditional_conditioning(num_samples, '')
		uc_full = {'c_concat': [c_cat], 'c_crossattn': [uc_cross]}
		if not do_full_sample:
			# encode (scaled latent)
			z_enc = sampler.stochastic_encode(
				z, torch.tensor([t_enc] * num_samples).to(model.device))
		else:
			z_enc = torch.randn_like(z)
		# decode it
		samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
								 unconditional_conditioning=uc_full, callback=callback)
		x_samples_ddim = model.decode_first_stage(samples)
		result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
		result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
	return [depth_image] + [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

def pad_image(input_image):
	pad_w, pad_h = np.max(((2, 2), np.ceil(
		np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
	im_padded = Image.fromarray(
		np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
	w, h = im_padded.size
	if w == h:
		return im_padded
	elif w > h:
		new_image = Image.new(im_padded.mode, (w, w), (0, 0, 0))
		new_image.paste(im_padded, (0, (w - h) // 2))
		return new_image
	else:
		new_image = Image.new(im_padded.mode, (h, h), (0, 0, 0))
		new_image.paste(im_padded, ((h - w) // 2, 0))
		return new_image


@app.route('/depth/predict', methods=['OPTIONS', 'POST'])
def predict():
	if (request.method == 'OPTIONS'):
		print('got options 1')
		response = Response()
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		print('got options 2')
		return response

	body = request.get_data()
	input_image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)

	prompt = request.args.get('prompt')
	# steps = request.args.get('steps')
	# num_samples = request.args.get('num_samples')
	# scale = request.args.get('scale')
	# seed = request.args.get('seed')
	# eta = request.args.get('eta')
	# strength = request.args.get('strength')
	steps = 50
	num_samples = 1
	scale = 9.0
	seed = 123123123
	eta = 0.0
	strength = 0.8
	#&steps=50&num_samples=1&scale=9.0&seed=123123123&eta=0.0&strength=0.8

	num_samples = 1
	#init_image = input_image
	#image = pad_image(input_image)  # resize to integer multiple of 32
	# image = input_image.resize((512, 512))
	dim = (512, 512)
	image = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
	sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
	assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
	do_full_sample = strength == 1.
	t_enc = min(int(strength * steps), steps-1)
	result = paint(
		sampler=sampler,
		image=image,
		prompt=prompt,
		t_enc=t_enc,
		seed=seed,
		scale=scale,
		num_samples=num_samples,
		callback=None,
		do_full_sample=do_full_sample
	)
	# return result
	# d_result = cv2.imencode('.png', result[1])[1].tobytes()
	img_byte_arr = io.BytesIO()
	result[1].save(img_byte_arr, format='PNG')
	img_byte_arr = img_byte_arr.getvalue()
	response = Response(img_byte_arr, headers={'Content-Type':'image/png'})
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Headers'] = '*'
	response.headers['Access-Control-Allow-Methods'] = '*'
	response.headers['Access-Control-Expose-Headers'] = '*'
	response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
	response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
	response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
	return response


if __name__ == '__main__':
	depth_model = 'configs/stable-diffusion/v2-midas-inference.yaml'
	checkpoints = 'configs/stable-diffusion/512-depth-ema.ckpt'
	sampler = initialize_model(depth_model, checkpoints)

	app.run(
		host='0.0.0.0',
		port=8081,
		threaded=True,
		debug=False
	)
