#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <vector>

// Load a grayscale image
cv::Mat loadImage(const std::string& imageFile) {
    cv::Mat img = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Couldn't open image " << imageFile << ".\n";
        exit(-1);
    }
    return img;
}


std::vector<cv::Mat> generateNoiseFrames_2(int width, int height, int numFrames, bool applyFilter) {
    std::vector<cv::Mat> noiseFrames;
    for (int i = 0; i < numFrames; ++i) {
        cv::Mat noise(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = static_cast<float>(rand() % 256);
                noise.at<uchar>(y, x) = static_cast<uchar>(value);
            }
        }
        if (applyFilter) {
            cv::GaussianBlur(noise, noise, cv::Size(5, 5), 0);
            noise = (noise - 128) * 2 + 128;
            
        }
        noiseFrames.push_back(noise);
    }
    return noiseFrames;
}



std::vector<cv::Mat> generateNoiseFrames_3(int width, int height, int numFrames, bool applyFilter) {
    std::vector<cv::Mat> noiseFrames;
    for (int i = 0; i < numFrames; ++i) {
        cv::Mat noise(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = static_cast<float>(rand() % 256);
                noise.at<uchar>(y, x) = static_cast<uchar>(value);
            }
        }
        if (applyFilter) {
            cv::GaussianBlur(noise, noise, cv::Size(5, 5), 0);

            // Transform the noise
            noise.convertTo(noise, CV_32F); // Convert to float for transformation
            noise -= 128; // Subtract 128
            noise *= 2;   // Multiply by 2
            noise += 128; // Add 128 back
            noise.convertTo(noise, CV_8UC1); // Convert back to 8-bit
        }
        noiseFrames.push_back(noise);
    }
    return noiseFrames;
}



// Generate grayscale noise frames
std::vector<cv::Mat> generateNoiseFrames(int width, int height, int numFrames) {
    std::vector<cv::Mat> noiseFrames;
    for (int i = 0; i < numFrames; ++i) {
        cv::Mat noise(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = static_cast<float>(rand() % 256);
                noise.at<uchar>(y, x) = static_cast<uchar>(value);
            }
        }
        noiseFrames.push_back(noise);
    }
    return noiseFrames;
}

// Create a parabolic lookup table
std::vector<uchar> createParabolicLUT() {
    std::vector<uchar> lut(256);
    for (int i = 0; i < 256; ++i) {
        float normalized = i / 255.0f;
        lut[i] = static_cast<uchar>(std::round(255.0f * normalized * normalized));
    }
    return lut;
}

// Apply the lookup table to an image
void applyLUT(const cv::Mat& src, cv::Mat& dst, const std::vector<uchar>& lut) {
    dst = src.clone();
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            dst.at<uchar>(y, x) = lut[src.at<uchar>(y, x)];
        }
    }
}

#define noise_level .5


void blendImagesAndNoise(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::Mat>& noiseFrames, 
                         cv::Mat& resultImg, const std::vector<uchar>& lut, float imageBlendWeight, 
                         float noiseWeight) {

    static int nf_index;
    // Use pre-generated noise frame
    cv::Mat noise = noiseFrames[nf_index];
    nf_index = nf_index + 1;
    nf_index = nf_index % noiseFrames.size();

    // Blend image1 and image2
    cv::Mat blendedImages;
    cv::addWeighted(img1, imageBlendWeight, img2, imageBlendWeight, 0.0, blendedImages);

    // Blend the result with noise
    cv::Mat finalBlendedImg;
    cv::addWeighted(blendedImages, 1.0f - noiseWeight, noise, noiseWeight, 0.0, finalBlendedImg);

    // Apply the parabolic lookup table
    applyLUT(finalBlendedImg, resultImg, lut);
}



int main() {
    std::string imageFile1 = "/home/jim/Desktop/PiTests/images/image.jpg"; // Path to your first image file
    std::string imageFile2 = "/home/jim/Desktop/PiTests/images/image2.jpg"; // Path to your second image file

    cv::Mat img1 = loadImage(imageFile1);
    cv::Mat img2 = loadImage(imageFile2);


    cv::Mat finalBlendedImg;
    cv::Mat blendedImages;
    cv::Mat transformedImg;

    const int numNoiseFrames = 30;
    // std::vector<cv::Mat> noiseFrames = generateNoiseFrames(img1.cols, img1.rows, numNoiseFrames);
     std::vector<cv::Mat> noiseFrames = generateNoiseFrames_2(img1.cols, img1.rows, numNoiseFrames, true);

    cv::namedWindow("Blended Image Playback", cv::WINDOW_AUTOSIZE);
    int noiseFrameIndex = 0;

    // Create the parabolic lookup table
    std::vector<uchar> lut = createParabolicLUT();


    
   while (true) {
        auto loopStartTime = std::chrono::steady_clock::now();

        // Use pre-generated noise frame
        // cv::Mat noise = noiseFrames[noiseFrameIndex];
        // noiseFrameIndex = (noiseFrameIndex + 1) % numNoiseFrames;

        // Blend image1 and image2 (1/2 each)
        
        // cv::addWeighted(img1, 0.5, img2, 0.5, 0.0, blendedImages);

        // // Blend the result with noise (1/2 each)

        // cv::addWeighted(blendedImages, 1-noise_level, noise, noise_level, 0.0, finalBlendedImg);

        // // Apply the parabolic lookup table
        
        // applyLUT(finalBlendedImg, transformedImg, lut);


        blendImagesAndNoise(img1, img2, noiseFrames, transformedImg, lut, .5, .5);



        transformedImg = transformedImg * 1.8;


        cv::imshow("Blended Image Playback", transformedImg);

        // Check for escape key press
        int key = cv::waitKey(1);
        if (key == 27) { // ASCII code for the escape key
            break;
        }

        auto loopEndTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = loopEndTime - loopStartTime;
        std::cout << "Loop duration: " << elapsed_seconds.count() << "s\n";
    }

    return 0;
}