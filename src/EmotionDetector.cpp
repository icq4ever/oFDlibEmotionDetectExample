#include "EmotionDetector.h"

EmotionDetector::EmotionDetector(){
	setup();
}
void EmotionDetector::setup(){
	shapeFileName 		= "shape_predictor_68_face_landmarks.dat";
	emotionFileName1 	= "classifiers/neutral_vs_happy.dat";
	emotionFileName2 	= "classifiers/neutral_vs_sad.dat";
	emotionFileName3 	= "classifiers/neutral_vs_surprise.dat";
	emotionFileName4 	= "classifiers/happy_vs_sad.dat";
	emotionFileName5 	= "classifiers/happy_vs_surprise.dat";
	emotionFileName6 	= "classifiers/sad_vs_surprise.dat";

	dlib::deserialize(ofToDataPath(shapeFileName, true)) >> sp;
	dlib::deserialize(ofToDataPath(emotionFileName1, true)) >> ep1;
	dlib::deserialize(ofToDataPath(emotionFileName2, true)) >> ep2;
	dlib::deserialize(ofToDataPath(emotionFileName3, true)) >> ep3;
	dlib::deserialize(ofToDataPath(emotionFileName4, true)) >> ep4;
	dlib::deserialize(ofToDataPath(emotionFileName5, true)) >> ep5;
	dlib::deserialize(ofToDataPath(emotionFileName6, true)) >> ep6;

//	detector = dlib::get_frontal_face_detector();
}

EmotionDetector::~EmotionDetector(){}


sample_type EmotionDetector::getAllAttributes(dlib::full_object_detection shape){
	sample_type sample;

	int l = 0;
	for(int j = 0; j < 68; j++){
		for(int k = 0; k < j; k++,l++){
			sample(l) = length(shape.part(j),shape.part(k));
			l++;
			sample(l) = slope(shape.part(j),shape.part(k));
		}
	}

	sample_type newSample = sample;
	return newSample;
}

void EmotionDetector::getEmotion(dlib::full_object_detection shape){
	sample_type sample;
	sample = getAllAttributes(shape);

	std::vector<double> prob;
	std::vector<double> emoProb;
	prob = svmMulticlass(sample);
	emoProb = probablityCalculator(prob);

	// get maximum probabiliry index from vector "prob"
	std::vector<double>::iterator result;
	result = std::max_element(emoProb.begin(), emoProb.begin()+4);
	int idx = result - emoProb.begin();

	switch(idx){
	case 0:
		std::cout <<"face is nutural" << std::endl;
		break;
	case 1 :
		std::cout << "face is happy" << std::endl;
		break;
	case 2:
		std::cout << "face is sad" << std::endl;
		break;
	case 3:
		std::cout << "face is surprise" << std::endl;
		break;
	default:
		std::cout << "unknown" << std::endl;
		break;
	}
}

void EmotionDetector::getEmotionFromImage(ofPixels pixels){
	std::cout << "get Emotion From Image" << std::endl;
	dlib::frontal_face_detector dt = dlib::get_frontal_face_detector();
	sample_type sample;

	std::vector<dlib::rectangle> faceRects = dt(pixels);
	faceRects = dt(pixels);

	if(faceRects.size() > 0) {
		dlib::full_object_detection feature = sp(pixels, faceRects[0]);

		int l = 0;
		for(int j = 0; j < 68; j++){
			for(int k = 0; k < j; k++,l++){
				sample(l) = length(feature.part(j),feature.part(k));
				l++;
				sample(l) = slope(feature.part(j),feature.part(k));
			}
		}

		sample_type newSample = sample;
		std::vector<double> prob;
		std::vector<double> emoProb;

		prob = svmMulticlass(newSample);
		emoProb = probablityCalculator(prob);
		
		for(int i=0; i<4; i++){
			std::cout << i << " : " << emoProb[i] << std::endl;
		}
		std::cout << std::endl;

		std::vector<double>::iterator result;
		result = std::max_element(emoProb.begin(), emoProb.begin() + 4);
		int idx = std::distance(emoProb.begin(), result);

		std::cout << "idx:" << idx << std::endl;
		switch(idx){
		case 0:
			std::cout <<"face is nutural" << std::endl;
			break;
		case 1 :
			std::cout << "face is happy" << std::endl;
			break;
		case 2:
			std::cout << "face is sad" << std::endl;
			break;
		case 3:
			std::cout << "face is surprise" << std::endl;
			break;
		default:
			std::cout << "unknown" << std::endl;
			break;
		}

//		std::cin.ignore();
	}
}



double EmotionDetector::length(dlib::point a, dlib::point b){
	int x1,y1,x2,y2;
	double dist;
	x1 = a.x();
	y1 = a.y();
	x2 = b.x();
	y2 = b.y();

	dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
	dist = std::sqrt(dist);
	return dist;
}
double EmotionDetector::slope(dlib::point a, dlib::point b){
	int x1,y1,x2,y2;

	x1 = a.x();
	y1 = a.y();
	x2 = b.x();
	y2 = b.y();
	if((x1-x2) == 0)
		if((y1-y2) > 0)
			return (M_PI/2);
		else
			return (-M_PI/2);
	else
		return atan(double(y1-y2))/(x1-x2);
}

std::vector<double> EmotionDetector::probablityCalculator(std::vector<double> P){
	std::vector<double> EmoProb(4);
	float e[4],temp;

	for(int i=0;i < 6;i++){
		P.push_back(1-P[i]);
	}

	e[0] = P[0]+P[1]+P[2];
	e[1] = P[3]+P[4]+P[6];
	e[2] = P[5]+P[7]+P[9];
	e[3] = P[8]+P[10]+P[11];

	int  i, j;
	int t[]={0,1,2,3};

	for(i=0;i<4;i++){
		for(j=i+1;j<4;j++){
			if(e[i]<e[j]){
				temp=e[i];
				e[i]=e[j];
				e[j]=temp;

				temp=t[i];
				t[i]=t[j];
				t[j]=temp;
			}
		}
	}

	e[0] = e[0]/3;
	e[1]= (1-e[0])*e[1]/3;
	e[2]=(1-e[0]-e[1])*e[2]/3;
	e[3]=(1-e[0]-e[1]-e[2]);

	for(i=0;i<4;i++){
		EmoProb[t[i]]=e[i];
	}

	return EmoProb;
}

std::vector<double> EmotionDetector::svmMulticlass(sample_type sample){
	std::vector<double> probs;
	probs.push_back(ep1(sample));
	probs.push_back(ep2(sample));
	probs.push_back(ep3(sample));
	probs.push_back(ep4(sample));
	probs.push_back(ep5(sample));
	probs.push_back(ep6(sample));

	return probs;
}
