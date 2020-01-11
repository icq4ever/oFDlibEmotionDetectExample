//
// Copyright (c) 2017 Christopher Baker <https://christopherbaker.net>
//
// SPDX-License-Identifier:	MIT
//


#include "ofApp.h"


void ofApp::setup()
{
    cam.setDeviceID(2);
	cam.setDesiredFrameRate(30);
    cam.setup(640, 360);
    // cam.initGrabber(720, 480);

    // We need a face detector. We will use this to get bounding boxes for
    // each face in an image.
    // dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    detector = dlib::get_frontal_face_detector();

    // Allocate some pixels.
    // ofPixels pixels;

    
    // Load an image.
    // ofLoadImage(pixels, "2007_007763.png");

    // Make the image bigger by a factor of two.  This is useful since
    // the face detector looks for faces that are about 80 by 80 pixels
    // or larger.  Therefore, if you want to find faces that are smaller
    // than that then you need to upsample the image as we do here by
    // calling pyramid_up().  So this will allow it to detect faces that
    // are at least 40 by 40 pixels in size.  We could call pyramid_up()
    // again to find even smaller faces, but note that every time we
    // upsample the image we make the detector run slower since it must
    // process a larger image.
    // dlib::pyramid_up(pixels);

    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    // faceRects = detector(pixels);

    // Set the image from the pixels.
    // image.setFromPixels(pixels);

    dlib::deserialize(ofToDataPath("shape_predictor_68_face_landmarks.dat", true)) >> sp;

//	ed.setup();
}

void ofApp::update(){
    dets.clear();
    shapes.clear();
    faceChips.clear();
    cam.update();
    if(cam.isFrameNew()){
        pixels = cam.getPixels();
        

        dets = detector(pixels);
        
        for(size_t j=0; j<dets.size(); ++j){
			dlib::full_object_detection shape = sp(pixels, dets[0]);
//			ed.getEmotion(shape);
            shapes.push_back(shape);
        }

		vector<dlib::chip_details> chipDetails = dlib::get_face_chip_details(shapes);

        dlib::array<ofPixels> face_chips;

        dlib::extract_image_chips(pixels, chipDetails, face_chips);

        for(auto& f: face_chips){
            faceChips.push_back(ofImage(f));
        }

        if(face_chips.size() > 0)   {
//			 ed.getEmotionFromImage(face_chips[0]);
            // cout << "oh" << endl;
        }

        image.setFromPixels(pixels);

        // Set the image from the pixels.
        // image.setFromPixels(pixels);

        if(keyCheck){
			faceChips[0].save("fff.jpg");
			ofPixels tPixels = face_chips[0];
			tPixels.resize(500, 500, OF_INTERPOLATE_NEAREST_NEIGHBOR);
			ed.getEmotionFromImage(tPixels);
            keyCheck = false;
        }
    }
}


void ofApp::draw()
{
    ofBackground(0);
    ofNoFill();
    ofSetColor(ofColor::white);

    ofPushMatrix();

    image.draw(0, 0);

    for (auto& shape: shapes){
        ofSetColor(ofColor::yellow);
        ofDrawRectangle(ofxDlib::toOf(shape.get_rect()));

        for(size_t i=0; i<shape.num_parts(); ++i){
            ofDrawCircle(ofxDlib::toOf(shape.part(i)), 2);
        }
    }

    float x = 0;
    float y = 100;
	// ofScale(0.5, 0.5)

    for(size_t i=0; i<faceChips.size(); ++i){
//		 if(faceChips.size() > 0)    y += faceChips[0].getHeight();
        auto & face = faceChips[i];

        if(i !=0 && x + face.getWidth() > ofGetWidth()){
            y += face.getHeight();
            x = ofGetHeight();
        }

        ofSetColor(ofColor::white);
		face.draw(x, y, 100, 100);

        x += face.getWidth();
    }

    ofPopMatrix();

    // ofScale(0.5, 0.5);
    ofDrawBitmapStringHighlight("Num. faces detected: " + ofToString(faceChips.size()), 14, 20);
}

void ofApp::keyPressed(int key){
    keyCheck = true;
}
