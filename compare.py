import pandas
import face_recognition
from os.path import join

class Comparison:

	def __init__(self, output_img, dataframe, person_id):
		self.output_img = output_img
		self.dataframe = dataframe
		self.person_id = person_id

	def compare(self):
		img_group_path = []

		# from the whole dataframe, take only the people with corresponding ID
		filt = (self.dataframe['identity'] == self.person_id)
		people_list = self.dataframe.loc[filt]['person'].tolist()

		for person in people_list:
			img_group_path.append(join('./dataset/CelebA/Img/img_align_celeba', person))

		# debug
		print(img_group_path[0])

		known_image = face_recognition.load_image_file(img_group_path[0])		# one person out of the group
		unknown_image = face_recognition.load_image_file(self.output_img)		# output generated

		# encoding
		biden_encoding = face_recognition.face_encodings(known_image)[0]		
		unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

		self.results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

	def getResult(self):
		return self.results