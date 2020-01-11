#pragma once

#include "ofConstants.h"
#include "ofGraphics.h"
#include "ofxDlib.h"

typedef dlib::matrix<double,4556,1> sample_type;

typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::probabilistic_decision_function<kernel_type> probabilistiopenc_funct_type;
typedef dlib::normalized_function<probabilistiopenc_funct_type> pfunct_type;

class EmotionDetector {
public:
	EmotionDetector();
	~EmotionDetector();
	void setup();
	sample_type getAllAttributes(dlib::full_object_detection shape);
	// std::vector<sample_type> getAllAttributes(ofPixels pixels);
	void getEmotion(dlib::full_object_detection shape);
	void getEmotionFromImage(ofPixels pixels);

	std::string shapeFileName;
	std::string emotionFileName1;
	std::string emotionFileName2;
	std::string emotionFileName3;
	std::string emotionFileName4;
	std::string emotionFileName5;
	std::string emotionFileName6;

	double length(dlib::point a, dlib::point b);
	double slope (dlib::point a, dlib::point b);
	std::vector<double> probablityCalculator(std::vector<double> P);

	std::vector<double> svmMulticlass(sample_type sample);

//	dlib::frontal_face_detector detector;

	dlib::shape_predictor sp;
	pfunct_type ep1;
	pfunct_type ep2;
	pfunct_type ep3;
	pfunct_type ep4;
	pfunct_type ep5;
	pfunct_type ep6;
};
