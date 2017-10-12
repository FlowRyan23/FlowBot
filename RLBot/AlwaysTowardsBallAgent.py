import tensorflow as tf
import atexit
from learning.Learner import tri_layer_ff_nn
from learning.Learner import recurrent_nn

# Optional Information. Fill out only if you wish.

# Your real name:
# Contact Email: flowrian.23@googlemail.com
# Can this bot's code be shared publicly (Default: No):
# Can non-tournament gameplay of this bot be displayed publicly (Default: No):


# This is the name that will be displayed on screen in the real time display!
BOT_NAME = "FlowBot"
NET_NAME = 'Flow_Bot_RNN_1M'

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# number of inputs/outputs
n_input = 41
n_output = 14


def get_input_vector_ff(gameTickPacket):
	# [BallX, -Y, -Z, BallRotX, -Y, -Z, BallVelX, -Y, -Z, BallAngVelX, -Y, -Z, BallAccX, -Y, -Z] entrys 0-14 (SelfInfo)
	# [SelfX, -Y, -Z, SelfRotX, -Y, -Z, SelfVelX, -Y, -Z, SelfAngVelX, -Y, -Z, SelfAccX, -Y, -Z, Boost] entrys 15-24 (PlayerInfo self)
	# [OppX, -Y, -Z, OppRotX, -Y, -Z, OppVelX, -Y, -Z, OppAngVelX, -Y, -Z, OppAccX, -Y, -Z, Boost] entrys 25-40 (PlayerInfo opponent)
	vector = [round(gameTickPacket.gameball.Location.X, 2),				# BallInfo
			  round(gameTickPacket.gameball.Location.Y, 2),
			  round(gameTickPacket.gameball.Location.Z, 2),
			  round(gameTickPacket.gameball.Rotation.Pitch, 2),
			  round(gameTickPacket.gameball.Rotation.Yaw, 2),
			  round(gameTickPacket.gameball.Rotation.Roll, 2),
			  round(gameTickPacket.gameball.Velocity.X, 2),
			  round(gameTickPacket.gameball.Velocity.Y, 2),
			  round(gameTickPacket.gameball.Velocity.Z, 2),
			  round(gameTickPacket.gameball.AngularVelocity.X, 2),
			  round(gameTickPacket.gameball.AngularVelocity.Y, 2),
			  round(gameTickPacket.gameball.AngularVelocity.Z, 2),
			  round(gameTickPacket.gameball.Acceleration.X, 2),
			  round(gameTickPacket.gameball.Acceleration.Y, 2),
			  round(gameTickPacket.gameball.Acceleration.Z, 2),
			  round(gameTickPacket.gamecars[0].Location.X, 2),			# PlayerInfo self (blue team)
			  round(gameTickPacket.gamecars[0].Location.Y, 2),
			  round(gameTickPacket.gamecars[0].Location.Z, 2),
			  round(gameTickPacket.gamecars[0].Rotation.Pitch, 2),
			  round(gameTickPacket.gamecars[0].Rotation.Yaw, 2),
			  round(gameTickPacket.gamecars[0].Rotation.Roll, 2),
			  round(gameTickPacket.gamecars[0].Velocity.X, 2),
			  round(gameTickPacket.gamecars[0].Velocity.Y, 2),
			  round(gameTickPacket.gamecars[0].Velocity.Z, 2),
			  round(gameTickPacket.gamecars[0].AngularVelocity.X, 2),
			  round(gameTickPacket.gamecars[0].AngularVelocity.Y, 2),
			  round(gameTickPacket.gamecars[0].AngularVelocity.Z, 2),
			  gameTickPacket.gamecars[0].Boost,
			  round(gameTickPacket.gamecars[1].Location.X, 2),			# PlayerInfo opponent (orange team)
			  round(gameTickPacket.gamecars[1].Location.Y, 2),
			  round(gameTickPacket.gamecars[1].Location.Z, 2),
			  round(gameTickPacket.gamecars[1].Rotation.Pitch, 2),
			  round(gameTickPacket.gamecars[1].Rotation.Yaw, 2),
			  round(gameTickPacket.gamecars[1].Rotation.Roll, 2),
			  round(gameTickPacket.gamecars[1].Velocity.X, 2),
			  round(gameTickPacket.gamecars[1].Velocity.Y, 2),
			  round(gameTickPacket.gamecars[1].Velocity.Z, 2),
			  round(gameTickPacket.gamecars[1].AngularVelocity.X, 2),
			  round(gameTickPacket.gamecars[1].AngularVelocity.Y, 2),
			  round(gameTickPacket.gamecars[1].AngularVelocity.Z, 2),
			  gameTickPacket.gamecars[1].Boost]

	return [vector]


def get_input_vector_rec(game_tick_packet):
	ff_vector = get_input_vector_ff(game_tick_packet)
	rec_vector = []

	# todo input_vector not in right shape: "ValueError: Cannot feed value of shape (1, 1, 41) for Tensor 'Placeholder:0', which has shape '(?, 41, 1)'"
	# [[[val], [val], ...]]
	for val in ff_vector[0]:
		rec_vector.append([val])

	return [rec_vector]


def restore_session():
	saver = tf.train.Saver()

	config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'GPU': 0})
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())

	nn_path = '../learning/Trained_NNs/' + NET_NAME + '.ckpt'
	saver.restore(sess, nn_path)

	return sess


def convert_stick_output(value):
	# temporary because nn_output is hard to deal with
	if value < 0:
		return 0
	return 32767


def convert_trigger_output(value):
	# temporary because nn_output is hard to deal with
	if value < 0:
		return 0
	return 32767


def convert_button_output(value):
	if value > 0:
		return 1
	return 0


class Agent:
	def __init__(self, team):
		self.team = team  # use self.team to determine what team you are. I will set to "blue" or "orange"
		# self.x, _, self.y = tri_layer_ff_nn()
		self.x, _, self.y = recurrent_nn()

		self.sess = restore_session()
		atexit.register(self.close_session)

	def get_output_vector(self, sharedValue):
		game_tick_packet = sharedValue.GameTickPacket
		# nn_input_vector = get_input_vector_ff(game_tick_packet)
		nn_input_vector = get_input_vector_rec(game_tick_packet)

		nn_output_vector = self.sess.run(self.y, feed_dict={self.x: nn_input_vector})[0]

		r_thumb_x = convert_stick_output(nn_output_vector[0])
		r_thumb_y = convert_stick_output(nn_output_vector[1])
		acc = convert_trigger_output(nn_output_vector[4])
		decc = convert_trigger_output(nn_output_vector[5])
		boost = convert_button_output(nn_output_vector[8])
		jump = convert_button_output(nn_output_vector[9])
		slide = convert_button_output(nn_output_vector[12])

		if acc >= decc:
			decc = 0
		else:
			acc = 0

		return_vector = [int(r_thumb_x), int(r_thumb_y), int(acc), int(decc), int(jump), int(boost), int(slide)]
		print('------------------------')
		print(nn_output_vector)
		print(return_vector)
		return return_vector

	def close_session(self):
		self.sess.close()
