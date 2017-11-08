#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
float PSNR(const cv::Mat &src1, const cv::Mat &src2){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if (src1.empty() || src2.empty()){
		CV_Error(CV_HeaderIsNull, "[qualityassessment::PSNR] image is empty");
	}
	if (src1.type() != src2.type()){
		CV_Error(CV_StsBadArg, "[qualityassessment::PSNR] both types of image do not match");
	}

	//////////////////////////////////////////////////////////////////////////
	Mat	src1f, src2f, tmp1f;
	if (src1.type() == CV_32FC1){
		src1f = src1;
		src2f = src2;
	}
	else{
		src1.convertTo(src1f, CV_32FC1);
		src2.convertTo(src2f, CV_32FC1);
	}
	// get error and return
	tmp1f = src1f - src2f;
	float	mse = (float)sum(tmp1f.mul(tmp1f))[0] / (float)tmp1f.total();
	return static_cast<float>(10.*log10(255.*255. / mse));
}
float HPSNR(const cv::Mat &src1, const cv::Mat &src2, const int ksize){

	//////////////////////////////////////////////////////////////////////////
	///// exceptions
	if (src1.empty() || src2.empty()){
		CV_Error(CV_HeaderIsNull, "[qualityassessment::HPSNR] image is empty");
	}
	if (src1.type() != src2.type()){
		CV_Error(CV_StsBadArg, "[qualityassessment::HPSNR] both types of image do not match");
	}

	//////////////////////////////////////////////////////////////////////////
	// get Gaussian kernel. Please check the def of getGaussianKernel() for the exact value of the sigma (standard deviation)
	Mat	coe1f = getGaussianKernel(ksize, -1, CV_32FC1);
	// get blurred images
	Mat	src11f, src21f;
	if (src1.type() == CV_32FC1){
		src11f = src1;
		src21f = src2;
	}
	else{
		src1.convertTo(src11f, CV_32FC1);
		src2.convertTo(src21f, CV_32FC1);
	}
	sepFilter2D(src11f, src11f, -1, coe1f, coe1f, Point(-1, -1), 0, BORDER_REFLECT_101);
	sepFilter2D(src21f, src21f, -1, coe1f, coe1f, Point(-1, -1), 0, BORDER_REFLECT_101);
	// get error
	return PSNR(src11f, src21f);
}
float SSIM(const cv::Mat &src1, const cv::Mat &src2)
{
	//////////////////////////////////////////////////////////////////////////
	// exception
	if (src1.empty() || src2.empty()){
		CV_Error(CV_HeaderIsNull, "[qualityassessment::SSIM] image is empty");
	}
	if (src1.cols != src2.cols || src1.rows != src2.rows){
		CV_Error(CV_StsBadArg, "[qualityassessment::SSIM] sizes of two images are not equal");
	}
	if (src1.type() != CV_8U || src2.type() != CV_8U){
		CV_Error(CV_BadNumChannels, "[qualityassessment::SSIM] image should be grayscale");
	}
	//////////////////////////////////////////////////////////////////////////

	const int L = 255;
	double C1 = (0.01*L)*(0.01*L);		//C1 = (K1*L)^2, K1=0.01, L=255(for 8-bit grayscale)
	double C2 = (0.03*L)*(0.03*L);		//C1 = (K2*L)^2, K2=0.03, L=255(for 8-bit grayscale)
	double C3 = C2 / 2.0;
	double mean_x = 0, mean_y = 0, mean2_x = 0, mean2_y = 0, STDx = 0, STDy = 0, variance_xy = 0;
	float SSIMresult = 0;

	//mean X, mean Y
	for (int i = 0; i < src1.rows; i++){
		for (int j = 0; j < src1.cols; j++){
			mean_x += src1.data[i*src1.cols + j];
			mean_y += src2.data[i*src2.cols + j];
			mean2_x += (src1.data[i*src1.cols + j] * src1.data[i*src1.cols + j]);
			mean2_y += (src2.data[i*src2.cols + j] * src2.data[i*src2.cols + j]);
		}
	}
	mean_x /= (src1.rows * src1.cols);
	mean_y /= (src2.rows * src2.cols);
	mean2_x /= (src1.rows * src1.cols);
	mean2_y /= (src2.rows * src2.cols);

	//STD X, STD Y
	STDx = sqrt(mean2_x - mean_x * mean_x);
	STDy = sqrt(mean2_y - mean_y * mean_y);

	//variance_xy
	for (int i = 0; i < src1.rows; i++){
		for (int j = 0; j < src1.cols; j++){
			variance_xy += (src1.data[i*src1.cols + j] - mean_x) * (src2.data[i*src2.cols + j] - mean_y);
		}
	}
	variance_xy /= (src1.rows * src1.cols);

	SSIMresult = static_cast<float>(((2 * mean_x*mean_y + C1) * (2 * variance_xy + C2)) / ((mean_x*mean_x + mean_y*mean_y + C1) * (STDx*STDx + STDy*STDy + C2)));

	// return result of SSIM
	return SSIMresult;
}
Mat order(double arr[8][8], Mat src)
{

	Mat dst(src.rows, src.cols, CV_8UC1,Scalar(0));
	for (int r = 0; r < src.rows; r++)
		for (int c = 0; c < src.cols; c++)
		{
			int r_of = r % 8;
			int c_of = c % 8;
			if (src.ptr<uchar>(r)[c]>(arr[r_of][c_of] * 255))
				dst.ptr<uchar>(r)[c] = 255;
			else
				dst.ptr<uchar>(r)[c] = 0;
		}
	return dst;
}
Mat diff(Mat mask, Mat src,int thr)
{
	Mat dst(src.rows+2, src.cols+2, CV_8UC1, Scalar(0));
	Mat imageROI = dst(Rect(1, 1, src.cols, src.rows));
	src.copyTo(imageROI, src);
	src.convertTo(src, CV_64FC1);
	dst.convertTo(dst, CV_64FC1);
	for (int r = 1; r < src.rows+1; r++)
		for (int c = 1; c < src.cols+1; c++)
		{
			if (dst.ptr<double>(r)[c] > thr)
			{
				double error_d = 255 - dst.ptr<double>(r)[c];
				dst.ptr<double>(r)[c + 1] -= (error_d*mask.ptr<double>(0)[2]);
				dst.ptr<double>(r + 1)[c - 1] -=(error_d*mask.ptr<double>(1)[0]);
				dst.ptr<double>(r + 1)[c] -= (error_d*mask.ptr<double>(1)[1]);
				dst.ptr<double>(r + 1)[c + 1] -= (error_d*mask.ptr<double>(1)[2]);
				dst.ptr<double>(r)[c] = 255;
			}
			else
			{
				double error_d = dst.ptr<double>(r)[c];
				dst.ptr<double>(r)[c + 1] += (error_d*mask.ptr<double>(0)[2]);
				dst.ptr<double>(r + 1)[c - 1] += (error_d*mask.ptr<double>(1)[0]);
				dst.ptr<double>(r + 1)[c] += (error_d*mask.ptr<double>(1)[1]);
				dst.ptr<double>(r + 1)[c + 1] += (error_d*mask.ptr<double>(1)[2]);
				dst.ptr<double>(r)[c] = 0;
			}
		}
	dst.convertTo(dst, CV_8UC1);
	imageROI = dst(Rect(1, 1, src.cols, src.rows));
	return imageROI;
}
Mat diff_1976(Mat mask, Mat src, int thr)
{
	Mat dst(src.rows + 4, src.cols + 4, CV_8UC1, Scalar(0));
	Mat imageROI = dst(Rect(2, 2, src.cols, src.rows));
	src.copyTo(imageROI, src);
	src.convertTo(src, CV_64FC1);
	dst.convertTo(dst, CV_64FC1);
	for (int r = 2; r < src.rows + 2; r++)
		for (int c = 2; c < src.cols + 2; c++)
		{
			double error_d;
			if (dst.ptr<double>(r)[c] > thr)
			{
				error_d = -(255 - dst.ptr<double>(r)[c]);
				dst.ptr<double>(r)[c] = 255;
			}
			else
			{
				error_d = dst.ptr<double>(r)[c];
				dst.ptr<double>(r)[c] = 0;
			}
			dst.ptr<double>(r)[c + 1] += (error_d*mask.ptr<double>(0)[3]);
			dst.ptr<double>(r)[c + 2] += (error_d*mask.ptr<double>(0)[4]);
			dst.ptr<double>(r + 1)[c - 2] += (error_d*mask.ptr<double>(1)[0]);
			dst.ptr<double>(r + 1)[c - 1] += (error_d*mask.ptr<double>(1)[1]);
			dst.ptr<double>(r + 1)[c] += (error_d*mask.ptr<double>(1)[2]);
			dst.ptr<double>(r + 1)[c + 1] += (error_d*mask.ptr<double>(1)[3]);
			dst.ptr<double>(r + 1)[c + 2] += (error_d*mask.ptr<double>(1)[4]);
			dst.ptr<double>(r + 2)[c - 2] += (error_d*mask.ptr<double>(2)[0]);
			dst.ptr<double>(r + 2)[c - 1] += (error_d*mask.ptr<double>(2)[1]);
			dst.ptr<double>(r + 2)[c] += (error_d*mask.ptr<double>(2)[2]);
			dst.ptr<double>(r + 2)[c + 1] += (error_d*mask.ptr<double>(2)[3]);
			dst.ptr<double>(r + 2)[c + 2] += (error_d*mask.ptr<double>(2)[4]);
		}
	dst.convertTo(dst, CV_8UC1);
	imageROI = dst(Rect(2, 2, src.cols, src.rows));
	return imageROI;
}
int main()
{
	Mat src = imread("image.png", CV_LOAD_IMAGE_GRAYSCALE);
	double classical_4[8][8] = { 0.567, 0.635, 0.608, 0.514, 0.424, 0.365, 0.392, 0.486, 0.847, 0.878, 0.910, 0.698, 0.153, 0.122, 0.090, 0.302, 0.820, 0.969, 0.941, 0.667, 0.180, 0.031, 0.059, 0.333, 0.725, 0.788, 0.757, 0.545, 0.275, 0.212, 0.243, 0.455, 0.424, 0.365, 0.392, 0.486, 0.567, 0.635, 0.608, 0.514, 0.153, 0.122, 0.090, 0.302, 0.847, 0.878, 0.910, 0.698, 0.180, 0.031, 0.059, 0.333, 0.820, 0.969, 0.941, 0.667, 0.275, 0.212, 0.243, 0.455, 0.725, 0.788, 0.757, 0.545 };
	double bayer_5[8][8] = { 0.513, 0.272, 0.724, 0.483, 0.543, 0.302, 0.694, 0.453, 0.151, 0.755, 0.091, 0.966, 0.181, 0.758, 0.121, 0.936, 0.634, 0.392, 0.574, 0.332, 0.664, 0.423, 0.604, 0.362, 0.060, 0.875, 0.211, 0.815, 0.030, 0.906, 0.241, 0.845, 0.543, 0.302, 0.694, 0.453, 0.513, 0.272, 0.724, 0.483, 0.181, 0.758, 0.121, 0.936, 0.151, 0.755, 0.091, 0.966, 0.664, 0.423, 0.604, 0.362, 0.634, 0.392, 0.574, 0.332, 0.030, 0.906, 0.241, 0.845, 0.060, 0.875, 0.211, 0.815 };
	Mat Floy = (Mat_<double>(2, 3) << 0, 0, ((double)7 / 16), ((double)3 / 16), ((double)5 / 16), ((double)1 / 16));
	Mat Jarv = (Mat_<double>(3, 5) << 0, 0, 0, ((double)7 / 48), ((double)5 / 48), ((double)3 / 48), ((double)5 / 48), ((double)7 / 48), ((double)5 / 48), ((double)3 / 48), ((double)1 / 48), ((double)3 / 48), ((double)5 / 48), ((double)3 / 48), ((double)1 / 48));
	Mat Stu = (Mat_<double>(3, 5) << 0, 0, 0, ((double)8 / 42), ((double)4 / 42), ((double)2 / 42), ((double)4 / 42), ((double)8 / 42), ((double)4 / 42), ((double)2 / 42), ((double)1 / 42), ((double)2 / 42), ((double)4 / 42), ((double)2 / 42), ((double)1 / 42));
	//for classical
	Mat ans_1 = order(classical_4, src);
	Mat ans_2 = order(bayer_5, src);
	Mat ans_3=diff(Floy, src,128);
	Mat ans_4=diff_1976(Jarv, src, 128);
	Mat ans_5 = diff_1976(Stu, src, 128);
	float qual_1 = HPSNR(src,ans_1,5);
	float qual_2 = HPSNR(src,ans_2,5);
	float qual_3 = HPSNR(ans_3, src,5);
	float qual_4 = HPSNR(ans_4, src,5);
	float qual_5 = HPSNR(ans_5, src, 5);
	float qua2_1 = SSIM(src, ans_1);
	float qua3_2 = SSIM(src, ans_2);
	float qua4_3 = SSIM(src,ans_3);
	float qua5_4 = SSIM(src,ans_4);
	float qua5_5 = SSIM(src,ans_5);
	return 0;
}
