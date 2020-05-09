import pandas
import face_recognition
from os.path import join

class Comparison:

	def __init__(self, output_img, dataframe, person_id):
		self.output_img = output_img
		self.dataframe = dataframe
		self.person_id = person_id
		self.results = {}

	def compare(self):
		img_group_path = []

		# from the whole dataframe, take only the people with corresponding ID
		filt = (self.dataframe['identity'] == self.person_id)
		people_list = self.dataframe.loc[filt]['person'].tolist()

		for person in people_list:
			img_group_path.append(join('./dataset/CelebA/Img/img_align_celeba', person))

		# debug
		# print(img_group_path[0])

		for known_person in img_group_path:
			known_image = face_recognition.load_image_file(known_person)
			unknown_image = face_recognition.load_image_file(self.output_img)

			# encoding
			biden_encoding_try = face_recognition.face_encodings(known_image)		
			unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

			if len(biden_encoding_try) > 0:
				biden_encoding = biden_encoding_try[0]
				comparison = face_recognition.compare_faces([biden_encoding], unknown_encoding)
				# insert element in a dictionary where known_person is key
				self.results[known_person] = comparison
			else:
				print("No faces found in the image!")

	def getResult(self):
		return self.results