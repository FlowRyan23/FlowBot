import tensorflow as tf
import atexit

# Optional Information. Fill out only if you wish.

# Your real name:
# Contact Email: flowrian.23@googlemail.com
# Can this bot's code be shared publicly (Default: No):
# Can non-tournament gameplay of this bot be displayed publicly (Default: No):


# This is the name that will be displayed on screen in the real time display!
BOT_NAME = "FlowBot"


def get_input_vector(gameTickPacket):
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


class Agent:
	def __init__(self, team):
		self.team = team  # use self.team to determine what team you are. I will set to "blue" or "orange"

	def get_output_vector(self, sharedValue):
		game_tick_packet = sharedValue.GameTickPacket

		

		return_vector = [int(r_thumb_x), int(r_thumb_y), int(acc), int(decc), int(jump), int(boost), int(slide)]
		print(return_vector)
		return return_vector

	def close_session(self):
		self.sess.close()
