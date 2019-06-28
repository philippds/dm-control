from dm_control import suite
import numpy as np
from PIL import Image
import subprocess

# Load one task:
env = suite.load(domain_name="humanoid", task_name="run")

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

# Keep time step count to be able to stop otherwise infinitely long running tasks
time_step_counter = 0

# reset frames folder
subprocess.call(['rm', '-rf', 'frames'])
subprocess.call(['mkdir', '-p', 'frames'])

while not time_step.last() and time_step_counter < 500:
	action = np.random.uniform(action_spec.minimum, action_spec.maximum, size = action_spec.shape)
	time_step = env.step(action)
	image_data = env.physics.render(height = 480, width = 480, camera_id = "back")
	img = Image.fromarray(image_data, 'RGB')
	img.save("frames/frame-%.10d.png" % time_step_counter)
	time_step_counter += 1
	print(time_step.reward, time_step.discount, time_step.observation)

# convert frames to video
subprocess.call(['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r', '30', '-pix_fmt', 'yuv420p', 'video_name.mp4'])
