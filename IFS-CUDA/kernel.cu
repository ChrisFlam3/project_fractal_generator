#include "nmmintrin.h" 
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sstream>
#include <fstream> 
#include <thread>
#include <mutex>
#include <random>
#include <iostream>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#define NK_IMPLEMENTATION
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"
using namespace std;
sf::Sprite screen;
sf::Texture tex;
std::mutex histogramMutex;
float boundedFunctionScale = 1;
std::random_device rd;
std::default_random_engine e(rd());
std::uniform_int_distribution<> d(0, 1);
float width_calculation(nk_handle handle, float height, const char *text, int len)
{
	sf::Font *type = (sf::Font*)handle.ptr;
	sf::Text temp;
	temp.setFont(*type);
	temp.setString(text);
	temp.setCharacterSize(height);
	float text_width = temp.findCharacterPos(len).x - temp.findCharacterPos(0).x;
	//float text_width = temp.getLocalBounds().width;

	return text_width;
}

void nuklearSfmlDrawRectFilled(const nk_command *cmd, sf::RenderWindow &window) {
	nk_command_rect_filled *p = (nk_command_rect_filled *)cmd;
	sf::RectangleShape rectangle;
	rectangle.setFillColor(sf::Color(p->color.r, p->color.g, p->color.b, p->color.a));
	rectangle.setSize(sf::Vector2f(p->w, p->h));
	rectangle.setPosition(p->x, p->y);
	window.draw(rectangle);
}
void nuklearSfmlDrawText(const nk_command *cmd, sf::RenderWindow &window) {
	nk_command_text *p = (nk_command_text *)cmd;
	sf::Font* font = (sf::Font*)p->font->userdata.ptr;
	sf::Text text;
	text.setFont(*font);
	text.setString(p->string);
	text.setCharacterSize(p->height);
	text.setFillColor(sf::Color(p->foreground.r, p->foreground.g, p->foreground.b, p->foreground.a));
	text.setPosition(sf::Vector2f(p->x, p->y));
	window.draw(text);

}
void nuklearSfmlDrawScissor(const nk_command *cmd, sf::RenderWindow &window) {

	nk_command_scissor *p = (nk_command_scissor *)cmd;
	glEnable(GL_SCISSOR_TEST);
	glScissor(
		(GLint)(p->x),
		(GLint)((window.getSize().y - (GLint)(p->y + p->h))),//bottom left corner
		(GLint)(p->w),
		(GLint)(p->h));
}
void nuklearSfmlDrawRectOutline(const nk_command *cmd, sf::RenderWindow &window) {
	nk_command_rect *p = (nk_command_rect *)cmd;
	sf::RectangleShape rect;
	rect.setSize(sf::Vector2f(p->w, p->h));
	rect.setPosition(sf::Vector2f(p->x, p->y));
	rect.setOutlineThickness(p->line_thickness);
	rect.setFillColor(sf::Color(0, 0, 0, 0));
	rect.setOutlineColor(sf::Color(p->color.r, p->color.g, p->color.b, p->color.a));
	window.draw(rect);
}
void nuklearSfmlDrawCircleFilled(const nk_command *cmd, sf::RenderWindow &window) {
	nk_command_circle_filled *p = (nk_command_circle_filled *)cmd;
	sf::CircleShape circle;
	circle.setRadius(p->h / 2);
	circle.setPosition(sf::Vector2f(p->x, p->y));
	circle.setFillColor(sf::Color(p->color.r, p->color.g, p->color.b, p->color.a));
	window.draw(circle);

}
void nuklearSfmlDrawTriangleFilled(const nk_command *cmd, sf::RenderWindow &window) {
	nk_command_triangle_filled *p = (nk_command_triangle_filled *)cmd;
	sf::ConvexShape convex;
	convex.setPointCount(3);
	convex.setPoint(0, sf::Vector2f(p->a.x, p->a.y));
	convex.setPoint(1, sf::Vector2f(p->b.x, p->b.y));
	convex.setPoint(2, sf::Vector2f(p->c.x, p->c.y));
	convex.setFillColor(sf::Color(p->color.r, p->color.g, p->color.b, p->color.a));
	window.draw(convex);

}

void eventsToGui(sf::Event *evt, nk_context* ctx) {
	if (evt->type == sf::Event::MouseButtonPressed || evt->type == sf::Event::MouseButtonReleased) {
		int down = evt->type == sf::Event::MouseButtonPressed;
		const int x = evt->mouseButton.x, y = evt->mouseButton.y;
		if (evt->mouseButton.button == sf::Mouse::Left)
			nk_input_button(ctx, NK_BUTTON_LEFT, x, y, down);
		if (evt->mouseButton.button == sf::Mouse::Middle)
			nk_input_button(ctx, NK_BUTTON_MIDDLE, x, y, down);
		if (evt->mouseButton.button == sf::Mouse::Right)
			nk_input_button(ctx, NK_BUTTON_RIGHT, x, y, down);

	}
	else if (evt->type == sf::Event::MouseMoved) {
		nk_input_motion(ctx, evt->mouseMove.x, evt->mouseMove.y);

	}
	else if (evt->type == sf::Event::KeyPressed || evt->type == sf::Event::KeyReleased) {
		int down = evt->type == sf::Event::KeyPressed;

		if (evt->key.code == sf::Keyboard::Backspace) {
			nk_input_key(ctx, NK_KEY_BACKSPACE, down);
			nk_input_key(ctx, NK_KEY_BACKSPACE, 0);
		}
		else if (evt->key.code == sf::Keyboard::Enter) 
			nk_input_key(ctx, NK_KEY_ENTER, down);

		if (down == 1)
			if (evt->key.code >= sf::Keyboard::Num0&&evt->key.code <= sf::Keyboard::Num9)
				nk_input_char(ctx, evt->key.code + 22);
			else if (evt->key.code == sf::Keyboard::Period)
				nk_input_char(ctx, 46);
			else if (evt->key.code == sf::Keyboard::Dash)
				nk_input_char(ctx, 45);
			else if (evt->key.code == sf::Keyboard::Space)
				nk_input_char(ctx, 32);
			else if (evt->key.code >= sf::Keyboard::A&&evt->key.code <= sf::Keyboard::Z) {
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
					nk_input_char(ctx, evt->key.code + 65);
				else
					nk_input_char(ctx, evt->key.code + 97);

			}
	}

}
template <typename T>
T mix(T x, T y, T a) {
	return x*(1 - a) + y*a;
}
template <typename T>
T clamp(T x, T minVal, T maxVal) {
	return min(max(x, minVal), maxVal);
}
double determinant(sf::Transform mat) {
	const float* matPointer = mat.getMatrix();
	return abs(matPointer[0] * matPointer[5] * matPointer[10] - matPointer[4] * matPointer[1] * matPointer[10]);
}
void operator /=(sf::Transform &mat, double x)
{
	mat = sf::Transform(mat.getMatrix()[0] / x, mat.getMatrix()[4] / x, mat.getMatrix()[12], mat.getMatrix()[1] / x, mat.getMatrix()[5] / x, mat.getMatrix()[13], mat.getMatrix()[2], mat.getMatrix()[6], mat.getMatrix()[10]);

}
//gaussian
float calculateKernel(float x, float sigma)
{
	return 0.39894*exp(-0.5*x*x / (sigma*sigma)) / sigma;
}
//Unlinear functions
void spherical(sf::Vector2f &point) {
	point = 1 / (pow(point.x, 2) + pow(point.y, 2))*point;
}
void swirl(sf::Vector2f &point) {
	float r2 = pow(point.x, 2) + pow(point.y, 2);
	point = sf::Vector2f(point.x*sin(r2) - point.y*cos(r2), point.x*cos(r2) + point.y*sin(r2));
}
void handkerchief(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(r*sin(theta + r), r*cos(theta - r));
}
void sinusoidal(sf::Vector2f &point) {
	point = sf::Vector2f(sin(point.x)*boundedFunctionScale, sin(point.y)*boundedFunctionScale);
}
void linear(sf::Vector2f &point) {

}
void horseshoe(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	point = sf::Vector2f(1 / r*(point.x - point.y)*(point.x + point.y), 1 / r * 2 * point.x*point.y);
}
void polar(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(theta / 3.1415*boundedFunctionScale, (r - 1)*boundedFunctionScale);

}
void disc(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(theta / 3.1415*sin(3.1415*r)*boundedFunctionScale, theta / 3.1415*cos(3.1415*r)*boundedFunctionScale);

}
void heart(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(r*sin(theta*r), r*(-cos(theta*r)));
}
void spiral(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(1 / r*(cos(theta) + sin(r)), 1 / r*(sin(theta) - cos(r)));
}
void hyperbolic(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(sin(theta) / r, r*cos(theta));
}
void diamond(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(sin(theta)*cos(r)*boundedFunctionScale, cos(theta)*sin(r)*boundedFunctionScale);
}
void ex(sf::Vector2f &point) {
	float r = sqrt(pow(point.x, 2) + pow(point.y, 2));
	float theta = atan2(point.x, point.y);
	float p0 = sin(theta + r);
	float p1 = cos(theta - r);
	point = sf::Vector2f(r*(pow(p0, 3) + pow(p1, 3)), r*(pow(p0, 3) - pow(p1, 3)));
}
void julia(sf::Vector2f &point) {

	float omega = d(e)* 3.1415;
	float sqrtr = sqrt(sqrt(pow(point.x, 2) + pow(point.y, 2)));
	float theta = atan2(point.x, point.y);
	point = sf::Vector2f(sqrtr*cos(theta / 2 + omega)*boundedFunctionScale, sqrtr*sin(theta / 2 + omega)*boundedFunctionScale);

}
//SIMD functions
inline void SIMDMatrix32Mul(float *mat, sf::Vector2f *point) {//After performance tests it seems that SFML or compiler vectorize matrix-vector multiplication itself
	float temp,temp2;
	__m128 p = _mm_set_ps(point->x, point->y, 1, 0);
	_mm_store_ss(&temp, _mm_dp_ps(_mm_set_ps(mat[0],mat[1],mat[2],0), p , 0xFF));
	_mm_store_ss(&temp2, _mm_dp_ps(_mm_set_ps(mat[3], mat[4], mat[5], 0), p, 0xFF));
	point->x = temp;
	point->y = temp2;
}
//CUDA
//CUDA math functions
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
	a.x *= b;
	a.y *= b;
}
inline __host__ __device__ void mulMatVec(float *data,int index,float2* point)//multiply 3x2 matrix by vector
{
	float temp;
	
	temp = data[index] * point->x + data[index + 1] * point->y + data[index + 2];
	point->y= data[index+3] * point->x + data[index + 4] * point->y + data[index + 5];
	point->x = temp;
	
}
inline __device__ float calculateKernel_d(float x, float sigma)
{
	return 0.39894*expf(-0.5*x*x / (sigma*sigma)) / sigma;
}
//CUDA variations functions
__device__ float boundedFunctionScale_d = 1;
inline __device__ void spherical(float2 &point, curandState& s) {
	point = 1 / (powf(point.x, 2) + powf(point.y, 2))*(point);

}
inline __device__ void swirl(float2 &point, curandState& s) {
	float r2 = powf(point.x, 2) + powf(point.y, 2);
	point = make_float2(point.x*sinf(r2) - point.y*cosf(r2), point.x*cosf(r2) + point.y*sinf(r2));
}
inline __device__ void handkerchief(float2 &point, curandState& s) {
	float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	point = make_float2(r*sinf(theta + r), r*cosf(theta - r));
}
inline __device__ void sinusoidal(float2 &point, curandState& s) {
	point = make_float2(sinf(point.x)*boundedFunctionScale_d, sinf(point.y)*boundedFunctionScale_d);
}
inline __device__ void linear(float2 &point, curandState& s) {

}
inline __device__ void horseshoe(float2 &point, curandState& s) {
	float r = sqrtf(pow(point.x, 2) + powf(point.y, 2));
	point = make_float2(1 / r*(point.x - point.y)*(point.x + point.y), 1 / r * 2 * point.x*point.y);
}
inline __device__ void polar(float2 &point, curandState& s) {
	float r = sqrtf(pow(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	point = make_float2(theta / 3.1415*boundedFunctionScale_d, (r - 1)*boundedFunctionScale_d);

}
inline __device__ void disc(float2 &point, curandState& s) {
	float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	point = make_float2(theta / 3.1415*sinf(3.1415*r)*boundedFunctionScale_d, theta / 3.1415*cosf(3.1415*r)*boundedFunctionScale_d);

}
inline __device__ void heart(float2 &point, curandState& s) {
	float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	point = make_float2(r*sinf(theta*r), r*(-cosf(theta*r)));
}
inline __device__ void spiral(float2 &point, curandState& s) {
	float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	point = make_float2(1 / r*(cosf(theta) + sinf(r)), 1 / r*(sinf(theta) - cosf(r)));
}
inline __device__ void hyperbolic(float2 &point, curandState& s) {
	float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	point = make_float2(sinf(theta) / r, r*cosf(theta));
}
inline __device__ void diamond(float2 &point, curandState& s) {
	float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	point = make_float2(sinf(theta)*cosf(r)*boundedFunctionScale_d, cosf(theta)*sinf(r)*boundedFunctionScale_d);
}
__device__ void ex(float2 &point, curandState& s) {
	float r = sqrtf(powf(point.x, 2) + powf(point.y, 2));
	float theta = atan2f(point.x, point.y);
	float p0 = sinf(theta + r);
	float p1 = cosf(theta - r);
	point = make_float2(r*(powf(p0, 3) + powf(p1, 3)), r*(powf(p0, 3) - powf(p1, 3)));
}
__device__ void julia(float2 &point, curandState& s) {

	float omega = (int)(curand_uniform(&s)+0.5)* 3.1415;
	float sqrtr = sqrtf(sqrtf(powf(point.x, 2) + powf(point.y, 2)));
	float theta = atan2f(point.x, point.y);
	point = make_float2(sqrtr*cosf(theta / 2 + omega)*boundedFunctionScale_d, sqrtr*sinf(theta / 2 + omega)*boundedFunctionScale_d);

}
__device__ void(*deviceFunctionArray[14])(float2&, curandState&) = { spherical,swirl,handkerchief,sinusoidal,linear,horseshoe,polar,disc,heart,spiral,hyperbolic,diamond,ex,julia };//function pointers for device

//CUDA main algorithm
//pass1 0-5 current
//pass1 6 offset for random generator
//pass1 7 max

//pass2 0-5 probability
//pass2 6-11 mat1
//pass2 12-17 mat2
//pass2 18-23 mat3
//pass2 24-29 mat4
//pass2 30-35 mat5
//pass2 36-41 mat6
//pass2 42-47 color
//pass2 48 zoom
//pass2 49 boundedFunctionScale
//pass2 50+ postTransforms

//pass3 0-104 857 599 histogram count
//pass3 104 857 600-209 715 199 histogram color

__global__ void cudaAcceleratedHistogram(int* pass1,float* pass2,float* pass3) {
	boundedFunctionScale_d = pass2[49];
	curandState s;
	curand_init(blockIdx.x * 1024 + threadIdx.x+pass1[6], 0, 0, &s);

	float2 point = make_float2(0, 0);
	int temp, temp2;
	float cad = 0;
	for (int i = 0; i <1000; i++)
	{

		float r = curand_uniform(&s);
		
		

		if (r <= pass2[0]) { mulMatVec(pass2,6,&point); cad = (cad + pass2[42]) / 2; deviceFunctionArray[pass1[0]](point,s); mulMatVec(pass2, 50, &point);}
		else if (r <= pass2[1]) { mulMatVec(pass2, 12, &point); cad = (cad + pass2[43]) / 2; deviceFunctionArray[pass1[1]](point,s); mulMatVec(pass2, 56, &point);}
		else if (r <=pass2[2]) { mulMatVec(pass2, 18, &point); cad = (cad + pass2[44]) / 2; deviceFunctionArray[pass1[2]](point,s); mulMatVec(pass2, 62, &point);}
		else if (r <=pass2[3]) { mulMatVec(pass2, 24, &point) ; cad = (cad + pass2[45]) / 2; deviceFunctionArray[pass1[3]](point,s); mulMatVec(pass2, 68, &point);}
		else if (r <=pass2[4]) { mulMatVec(pass2, 30, &point); cad = (cad + pass2[46]) / 2; deviceFunctionArray[pass1[4]](point,s); mulMatVec(pass2, 74, &point);}
		else { mulMatVec(pass2, 36, &point); cad = (cad + pass2[47]) / 2; deviceFunctionArray[pass1[5]](point,s); mulMatVec(pass2, 80, &point);}
		
		
		
		
		if (i > 20) {
			if (abs(point.x) >= (5 * pass2[48])-0.001 || abs(point.y) >= (5 * pass2[48])-0.001)//GPU rounding makes exacly comp useless, need some space
				continue;
			temp = (int)((point.x + 5 * pass2[48]) * (1024 / pass2[48]));
			temp2 = (int)((point.y + 5 * pass2[48]) * (1024 / pass2[48]));
			temp2 = temp * 10240 + temp2;
			pass3[temp2]= (int)(pass3[temp2]+1);
			if (pass3[temp2] > pass1[7])
				atomicExch(&pass1[7],pass3[temp2]);
			pass3[104857600 + temp2] = (pass3[104857600 + temp2] + cad) / 2;

		}
	}
}

__global__ void cudaAcceleratedSupersampling(float* histogram, float* image,int* max) {

	float freq = 0;
	float color = 0;
	float alpha = 0;
	int y = blockIdx.x % 2*1024 + threadIdx.x;
	int i = blockIdx.x / 2;
			for (int a = 0; a < 5; a++) {
				for (int b = 0; b < 5; b++) {
					freq = histogram[5 * y * 10240 + b * 10240 + (5 * i + a)];
					color = histogram[5 * y * 10240 + b * 10240 + (5 * i + a) + 104857600];
					if (freq != 0)
						alpha += color*powf((log10f(freq) / log10f(*max)), 1 / 2.2);
					image[y * 2048 + i+ 4194304] += histogram[5 * y * 10240 + b * 10240 + (5 * i + a)];
				}
			}
			alpha /= 25;

			image[y * 2048 + i] = alpha;

		
}

__global__ void cudaAcceleratedGaussianBlur(float *image,int *gauss) {
	int x = blockIdx.x % 2 * 1024 + threadIdx.x;
	int y = blockIdx.x / 2;
	float kernel[20];

			int kernelWidth = *gauss / powf(image[x*2048+y+ 4194304], 0.4);
			if (kernelWidth >9|| kernelWidth <0)
				kernelWidth = *gauss;
			if (kernelWidth == 0)
				return;

			int kernelSize = kernelWidth * 2 + 1;
			float sigma = 6.;
			float Z = 0.0;
			float final_colour = 0;

			for (int j = 0; j <= kernelWidth; ++j)
			{
				kernel[kernelWidth + j] = kernel[kernelWidth - j] = calculateKernel_d(float(j), sigma);
			}

			for (int j = 0; j < kernelSize; ++j)
			{
				Z += kernel[j];
			}


			for (int i = -kernelWidth; i <= kernelWidth; ++i)
			{
				for (int j = -kernelWidth; j <= kernelWidth; ++j)
				{
					if (x + i < 0 || y + j < 0 || x + 1>2047 || y + j>2047)
						continue;
					final_colour += kernel[kernelWidth + j] * kernel[kernelWidth + i] * image[x*2048 + i*2048+y + j];

				}
			}
			final_colour /= (Z*Z);
			image[x * 2048 + y] = final_colour;
			
}

void calculateHistogramPart(float(*affineTransforms)[8], int current[], int affineNum, long long iterations, float(*histogram)[10241][2], float zoom, float(*postTransforms)[6]) {

	sf::Transform *affineMatrix = new sf::Transform[6];
	sf::Transform *postMatrix = new sf::Transform[6];
	for (int i = 0; i < affineNum; i++) {
		affineMatrix[i] = sf::Transform(affineTransforms[i][0], affineTransforms[i][1], affineTransforms[i][2], affineTransforms[i][3], affineTransforms[i][4], affineTransforms[i][5], 0, 0, 1);
		postMatrix[i] = sf::Transform(postTransforms[i][0], postTransforms[i][1], postTransforms[i][2], postTransforms[i][3], postTransforms[i][4], postTransforms[i][5], 0, 0, 1);
	}
	void(*unlinearPointer[14])(sf::Vector2f &point) = { spherical,swirl,handkerchief,sinusoidal,linear,horseshoe,polar,disc,heart,spiral,hyperbolic,diamond,ex,julia };//array of pointers to functions

	int max = 0;
	sf::Vector2f point = sf::Vector2f(0, 0);//be aware with translation 0 in matrix can cause division by 0
	float cad = 0;

	std::random_device rd;
	std::default_random_engine e(rd());
	std::uniform_real_distribution<> d(0.0, 1.0);

	float probability0 = affineTransforms[0][7];
	float probability1 = probability0 + affineTransforms[1][7];
	float probability2 = probability1 + affineTransforms[2][7];
	float probability3 = probability2 + affineTransforms[3][7];
	float probability4 = probability3 + affineTransforms[4][7];
	float probability5 = probability4 + affineTransforms[5][7];



	int temp, temp2;
	for (long long i = 0; i < iterations; i++)
	{

		float result = d(e);


		if (result < probability0) { point = affineMatrix[0] * point; cad = (cad + affineTransforms[0][6]) / 2; unlinearPointer[current[0]](point); point = postMatrix[0] * point;}
		else if (result < probability1) { point = affineMatrix[1] * point; cad = (cad + affineTransforms[1][6]) / 2; unlinearPointer[current[1]](point); point = postMatrix[1] * point;}
		else if (result <probability2) { point = affineMatrix[2] * point; cad = (cad + affineTransforms[2][6]) / 2; unlinearPointer[current[2]](point); point = postMatrix[2] * point;}
		else if (result <probability3) {point = affineMatrix[3] * point; cad = (cad + affineTransforms[3][6]) / 2; unlinearPointer[current[3]](point); point = postMatrix[3] * point;}
		else if (result <probability4) { point = affineMatrix[4] * point; cad = (cad + affineTransforms[4][6]) / 2; unlinearPointer[current[4]](point); point = postMatrix[4] * point;}
		else { point = affineMatrix[5] * point; cad = (cad + affineTransforms[5][6]) / 2; unlinearPointer[current[5]](point); point = postMatrix[5] * point;}

		if (i > 20) {
			if (abs(point.x) > 5 * zoom || abs(point.y) > 5 * zoom)
				continue;
			temp = (unsigned int)((point.x + 5 * zoom) * (1024 / zoom));
			temp2 = (unsigned int)((point.y + 5 * zoom) * (1024 / zoom));
			histogram[temp][temp2][0] += 1;
			//if (histogram[(int)((point.x + 5) * 1024)][(int)((point.y + 5) * 1024)][0] > max)
			//max = histogram[(int)((point.x + 5) * 1024)][(int)((point.y + 5) * 1024)][0];
			histogram[temp][temp2][1] = (histogram[temp][temp2][1] + cad) / 2;

		}

	}




}
void renderToFile(int renderType, float(*affineTransforms)[8], int current[], bool unlinearNum, int affineNum, float zoom, int gauss, int iterationsInMld, float(*postTransforms)[6]) {
	long long iterations = iterationsInMld * (long long)1000000000;
	if (renderType == 0) {
		if (unlinearNum == 0) {
			for (int i = 1; i<6; i++)
				current[i] = current[0];
		}
		sf::Transform *affineMatrix = new sf::Transform[6];
		sf::Transform *postMatrix = new sf::Transform[6];
		for (int i = 0; i < affineNum; i++) {
			affineMatrix[i] = sf::Transform(affineTransforms[i][0], affineTransforms[i][1], affineTransforms[i][2], affineTransforms[i][3], affineTransforms[i][4], affineTransforms[i][5], 0, 0, 1);
			postMatrix[i] = sf::Transform(postTransforms[i][0], postTransforms[i][1], postTransforms[i][2], postTransforms[i][3], postTransforms[i][4], postTransforms[i][5], 0, 0, 1);
			//affineMatrix[i] /= mix(1.0, determinant(affineMatrix[i]), clamp(determinant(affineMatrix[i])*3.0 - 2.0, 0.0, 1.0));
			//affineTransforms[i][0] = affineMatrix[i].getMatrix()[0];
			//affineTransforms[i][1] = affineMatrix[i].getMatrix()[4];
			//affineTransforms[i][2] = affineMatrix[i].getMatrix()[12];
			//affineTransforms[i][3] = affineMatrix[i].getMatrix()[1];
			//affineTransforms[i][4] = affineMatrix[i].getMatrix()[5];
			//affineTransforms[i][5] = affineMatrix[i].getMatrix()[13];

		}
		void(*unlinearPointer[14])(sf::Vector2f &point) = { spherical,swirl,handkerchief,sinusoidal,linear,horseshoe,polar,disc,heart,spiral,hyperbolic,diamond,ex,julia };//array of pointers to functions

		int max = 0;
		sf::Vector2f point = sf::Vector2f(0, 0);//be aware with translation 0 in matrix can cause division by 0
		float cad = 0;
		auto histogram = new float[10241][10241][2]();
		std::random_device rd;
		std::default_random_engine e(rd());
		std::uniform_real_distribution<> d(0.0, 1.0);

		float probability0 = affineTransforms[0][7];
		float probability1 = probability0 + affineTransforms[1][7];
		float probability2 = probability1 + affineTransforms[2][7];
		float probability3 = probability2 + affineTransforms[3][7];
		float probability4 = probability3 + affineTransforms[4][7];
		float probability5 = probability4 + affineTransforms[5][7];

		int temp, temp2;
		for (long long i = 0; i < iterations; i++)
		{
			float result = d(e);


			if (result < probability0) { point = affineMatrix[0] * point; cad = (cad + affineTransforms[0][6]) / 2; unlinearPointer[current[0]](point); point = postMatrix[0] * point;}
			else if (result < probability1) { point = affineMatrix[1] * point; cad = (cad + affineTransforms[1][6]) / 2; unlinearPointer[current[1]](point); point = postMatrix[1] * point;}
			else if (result <probability2) { point = affineMatrix[2] * point; cad = (cad + affineTransforms[2][6]) / 2; unlinearPointer[current[2]](point); point = postMatrix[2] * point;}
			else if (result <probability3) { point = affineMatrix[3] * point; cad = (cad + affineTransforms[3][6]) / 2; unlinearPointer[current[3]](point); point = postMatrix[3] * point;}
			else if (result <probability4) { point = affineMatrix[4] * point; cad = (cad + affineTransforms[4][6]) / 2; unlinearPointer[current[4]](point); point = postMatrix[4] * point;}
			else { point = affineMatrix[5] * point; cad = (cad + affineTransforms[5][6]) / 2; unlinearPointer[current[5]](point); point = postMatrix[5] * point;}

			if (i > 20) {
				if (abs(point.x) > 5 * zoom || abs(point.y) > 5 * zoom)
					continue;
				temp = (unsigned int)((point.x + 5 * zoom) * (1024 / zoom));
				temp2 = (unsigned int)((point.y + 5 * zoom) * (1024 / zoom));
				histogram[temp][temp2][0] += 1;
				if (histogram[temp][temp2][0] > max)
					max = histogram[temp][temp2][0];
				histogram[temp][temp2][1] = (histogram[temp][temp2][1] + cad) / 2;
			}
		}

		sf::Image img;
		img.create(2048, 2048);
		float freq = 0;
		float color = 0;
		float alpha = 0;
		auto freqArchive = new float[2048][2048];
		for (int i = 0; i < 2048; i++)
			for (int y = 0; y < 2048; y++) {
				for (int a = 0; a < 5; a++) {
					for (int b = 0; b < 5; b++) {
						freq = histogram[5 * y + b][5 * i + a][0];
						color = histogram[5 * y + b][5 * i + a][1];
						if (freq != 0)
							alpha += color*pow((log10(freq) / log10(max)), 1 / 2.2);
						freqArchive[y][i] += histogram[5 * y + b][5 * i + a][0];
					}
				}
				alpha /= 25;

				sf::Color col(alpha * 255, alpha * 255, alpha * 255, 255);
				img.setPixel(y, i, col);

			}
		//gaussian blur with density estimation
		float kernel[20];
		for (int y = 0; y<2048; y++)
			for (int x = 0; x < 2048; x++) {
				int kernelWidth = gauss / pow(freqArchive[x][y], 0.4);
				if (kernelWidth < 0)
					kernelWidth = gauss;
				if (kernelWidth == 0)
					continue;

				int kernelSize = kernelWidth * 2 + 1;
				//float* kernel = new float[kernelSize + 1];
				float sigma = 6.;
				float Z = 0.0;
				float final_colour = 0;

				for (int j = 0; j <= kernelWidth; ++j)
				{
					kernel[kernelWidth + j] = kernel[kernelWidth - j] = calculateKernel(float(j), sigma);
				}

				for (int j = 0; j < kernelSize; ++j)
				{
					Z += kernel[j];
				}


				for (int i = -kernelWidth; i <= kernelWidth; ++i)
				{
					for (int j = -kernelWidth; j <= kernelWidth; ++j)
					{
						if (x + i < 0 || y + j < 0 || x + 1>2047 || y + j>2047)
							continue;
						final_colour += kernel[kernelWidth + j] * kernel[kernelWidth + i] * img.getPixel(x + i, y + j).r;

					}
				}
				final_colour /= (Z*Z);
				sf::Color f(final_colour, final_colour, final_colour, 255);
				img.setPixel(x, y, f);


			}

		img.saveToFile("ifs.png");

		delete[] histogram;
		delete[] freqArchive;
	}
	else if (renderType == 1) {



		if (unlinearNum == 0) {
			for (int i = 1; i<6; i++)
				current[i] = current[0];
		}
		sf::Transform *affineMatrix = new sf::Transform[6];
		sf::Transform *postMatrix = new sf::Transform[6];
		for (int i = 0; i < affineNum; i++) {
			affineMatrix[i] = sf::Transform(affineTransforms[i][0], affineTransforms[i][1], affineTransforms[i][2], affineTransforms[i][3], affineTransforms[i][4], affineTransforms[i][5], 0, 0, 1);
			//affineMatrix[i] /= mix(1.0, determinant(affineMatrix[i]), clamp(determinant(affineMatrix[i])*3.0 - 2.0, 0.0, 1.0));
			//affineTransforms[i][0] = affineMatrix[i].getMatrix()[0];
			//affineTransforms[i][1] = affineMatrix[i].getMatrix()[4];
			//affineTransforms[i][2] = affineMatrix[i].getMatrix()[12];
			//affineTransforms[i][3] = affineMatrix[i].getMatrix()[1];
			//affineTransforms[i][4] = affineMatrix[i].getMatrix()[5];
			//affineTransforms[i][5] = affineMatrix[i].getMatrix()[13];

		}
		auto histogram = new float[10241][10241][2]();
		std::thread t1(calculateHistogramPart, affineTransforms, current, affineNum, iterations / 3, histogram, zoom,postTransforms);
		std::thread t2(calculateHistogramPart, affineTransforms, current, affineNum, iterations / 3, histogram, zoom, postTransforms);
		std::thread t3(calculateHistogramPart, affineTransforms, current, affineNum, iterations / 3, histogram, zoom, postTransforms);
		t1.join();
		t2.join();
		t3.join();
		int max = 0;
		for (int i = 0; i<10240; i++)
			for (int y = 0; y < 10240; y++) {
				if (histogram[y][i][0] > max)
					max = histogram[y][i][0];
			}



		sf::Image img;
		img.create(2048, 2048);
		float freq = 0;
		float color = 0;
		float alpha = 0;
		auto freqArchive = new float[2048][2048];
		for (int i = 0; i < 2048; i++)
			for (int y = 0; y < 2048; y++) {
				for (int a = 0; a < 5; a++) {
					for (int b = 0; b < 5; b++) {
						freq = histogram[5 * y + b][5 * i + a][0];
						color = histogram[5 * y + b][5 * i + a][1];
						if (freq != 0)
							alpha += color*pow((log10(freq) / log10(max)), 1 / 2.2);
						freqArchive[y][i] += histogram[5 * y + b][5 * i + a][0];
					}
				}
				alpha /= 25;

				sf::Color col(alpha * 255, alpha * 255, alpha * 255, 255);
				img.setPixel(y, i, col);

			}
		//gaussian blur with density estimation
		float kernel[20];
		for (int y = 0; y<2048; y++)
			for (int x = 0; x < 2048; x++) {
				int kernelWidth = gauss / pow(freqArchive[x][y], 0.4);
				if (kernelWidth < 0)
					kernelWidth = gauss;
				if (kernelWidth == 0)
					continue;

				int kernelSize = kernelWidth * 2 + 1;
				//float* kernel = new float[kernelSize + 1];
				float sigma = 6.;
				float Z = 0.0;
				float final_colour = 0;

				for (int j = 0; j <= kernelWidth; ++j)
				{
					kernel[kernelWidth + j] = kernel[kernelWidth - j] = calculateKernel(float(j), sigma);
				}

				for (int j = 0; j < kernelSize; ++j)
				{
					Z += kernel[j];
				}


				for (int i = -kernelWidth; i <= kernelWidth; ++i)
				{
					for (int j = -kernelWidth; j <= kernelWidth; ++j)
					{
						if (x + i < 0 || y + j < 0 || x + 1>2047 || y + j>2047)
							continue;
						final_colour += kernel[kernelWidth + j] * kernel[kernelWidth + i] * img.getPixel(x + i, y + j).r;

					}
				}
				final_colour /= (Z*Z);
				sf::Color f(final_colour, final_colour, final_colour, 255);
				img.setPixel(x, y, f);

			}

		img.saveToFile("ifs.png");

		delete[] histogram;


	}
else if (renderType == 2) {//CUDA acceleration
	
	std::random_device rd;
	std::default_random_engine e(rd());
	std::uniform_real_distribution<> d(0.0, 1.0);

	//memory allocation
	int *pass1_d;
	float *pass2_d;
	float *pass3_d;
	int *pass1_h;
	float *pass2_h;
	//device allocation
	cudaMalloc(&pass1_d, 8 * sizeof(int));
	cudaMalloc(&pass2_d, 87 * sizeof(float));
	cudaMalloc(&pass3_d, 209715200 * sizeof(float));
	cudaMemset(&pass3_d, 0, 209715200 * sizeof(float));

	//host allocation
	pass1_h = (int*)malloc(8 * sizeof(int));
	pass2_h = (float*)malloc(86 * sizeof(float));




	//pre-calculations
	if (unlinearNum == 0) {
		for (int i = 1; i<6; i++)
			current[i] = current[0];
	}
	sf::Transform *affineMatrix = new sf::Transform[6];
	sf::Transform *postMatrix = new sf::Transform[6];
	for (int i = 0; i < affineNum; i++) {
		affineMatrix[i] = sf::Transform(affineTransforms[i][0], affineTransforms[i][1], affineTransforms[i][2], affineTransforms[i][3], affineTransforms[i][4], affineTransforms[i][5], 0, 0, 1);
		postMatrix[i] = sf::Transform(postTransforms[i][0], postTransforms[i][1], postTransforms[i][2], postTransforms[i][3], postTransforms[i][4], postTransforms[i][5], 0, 0, 1);
		//affineMatrix[i] /= mix(1.0, determinant(affineMatrix[i]), clamp(determinant(affineMatrix[i])*3.0 - 2.0, 0.0, 1.0));
		//affineTransforms[i][0] = affineMatrix[i].getMatrix()[0];
		//affineTransforms[i][1] = affineMatrix[i].getMatrix()[4];
		//affineTransforms[i][2] = affineMatrix[i].getMatrix()[12];
		//affineTransforms[i][3] = affineMatrix[i].getMatrix()[1];
		//affineTransforms[i][4] = affineMatrix[i].getMatrix()[5];
		//affineTransforms[i][5] = affineMatrix[i].getMatrix()[13];

		//data feed
		pass2_h[6*i+6] = affineMatrix[i].getMatrix()[0];
		pass2_h[6*i+1+6] = affineMatrix[i].getMatrix()[4];
		pass2_h[6*i+2+6] = affineMatrix[i].getMatrix()[12];
		pass2_h[6*i+3+6] = affineMatrix[i].getMatrix()[1];
		pass2_h[6*i+4+6] = affineMatrix[i].getMatrix()[5];
		pass2_h[6*i+5+6] = affineMatrix[i].getMatrix()[13];

		pass2_h[42+i] = affineTransforms[i][6];

		//post transform feed
		pass2_h[6 * i + 50] = affineMatrix[i].getMatrix()[0];
		pass2_h[6 * i + 1 + 50] = affineMatrix[i].getMatrix()[4];
		pass2_h[6 * i + 2 + 50] = affineMatrix[i].getMatrix()[12];
		pass2_h[6 * i + 3 + 50] = affineMatrix[i].getMatrix()[1];
		pass2_h[6 * i + 4 + 50] = affineMatrix[i].getMatrix()[5];
		pass2_h[6 * i + 5 + 50] = affineMatrix[i].getMatrix()[13];

	}

	

	//data feed
	pass2_h[0] = affineTransforms[0][7];
	pass2_h[1] = pass2_h[0] + affineTransforms[1][7];
	pass2_h[2] = pass2_h[1] + affineTransforms[2][7];
	pass2_h[3] = pass2_h[2] + affineTransforms[3][7];
	pass2_h[4] = pass2_h[3] + affineTransforms[4][7];
	pass2_h[5] = pass2_h[4] + affineTransforms[5][7];

	for (int i = 0; i < 6; i++)
		pass1_h[i] = current[i];
	pass2_h[48] = zoom;
	pass2_h[49] = boundedFunctionScale;
	cudaMemcpy(pass2_d, pass2_h, 86 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pass1_d, pass1_h, 7 * sizeof(int), cudaMemcpyHostToDevice);
	
	int blocks;
	int loops;
	if (iterationsInMld < 10) {
		blocks = iterations / 1024000;
		loops = 1;
	}
	else {
		blocks = 2500;
		loops = (double)iterations / 2560000000+0.7;
	}
	
	for (int i = 0; i < loops; i++) {
		pass1_h[6] = d(e)*1000;
		cudaMemcpy(&pass1_d[6], &pass1_h[6],  sizeof(int), cudaMemcpyHostToDevice);
		cudaAcceleratedHistogram << <blocks, 1024 >> > (pass1_d, pass2_d, pass3_d);
		cudaDeviceSynchronize();
	}



	float *image_d;
	cudaMalloc(&image_d, 8388608 * sizeof(float));
	cudaAcceleratedSupersampling << <4096, 1024 >> > (pass3_d, image_d,&pass1_d[7]);

	

	cudaFree(pass3_d);

	int *gauss_d;
	cudaMalloc(&gauss_d, sizeof(int));
	cudaMemcpy(gauss_d, &gauss, sizeof(int), cudaMemcpyHostToDevice);


	cudaAcceleratedGaussianBlur << <4096, 1024 >> > (image_d, gauss_d);

	float* image_h = (float*)malloc(8388608 * sizeof(float));
	cudaMemcpy(image_h, image_d, 8388608 * sizeof(float), cudaMemcpyDeviceToHost);
			
			sf::Image img;
			float alpha;
			img.create(2048, 2048);
			for (int i = 0; i < 2048; i++)
				for (int y = 0; y < 2048; y++) {
					alpha = image_h[y * 2048 + i];
					sf::Color col(alpha * 255, alpha * 255, alpha * 255, 255);
					img.setPixel(y, i, col);

				}
			
		
			
			img.saveToFile("ifs.png");


			//memory deallocation
			cudaFree(pass2_d);
			cudaFree(pass1_d);
			cudaFree(image_d);
			free(pass1_h);
			free(pass2_h);
			free(image_h);


	}
}
void renderToOverview(int renderType, float(*affineTransforms)[8], int current[], bool unlinearNum, int affineNum, sf::RenderWindow& window, float zoom, float(*postTransforms)[6]) {


	//if (renderType == 0) {

	if (unlinearNum == 0) {
		for (int i = 1; i<6; i++)
			current[i] = current[0];
	}

	sf::Transform *affineMatrix = new sf::Transform[6];
	sf::Transform *postMatrix = new sf::Transform[6];
	for (int i = 0; i < affineNum; i++) {
		affineMatrix[i] = sf::Transform(affineTransforms[i][0], affineTransforms[i][1], affineTransforms[i][2], affineTransforms[i][3], affineTransforms[i][4], affineTransforms[i][5], 0, 0, 1);
		postMatrix[i] = sf::Transform(postTransforms[i][0], postTransforms[i][1], postTransforms[i][2], postTransforms[i][3], postTransforms[i][4], postTransforms[i][5],0,0,1);
		//affineMatrix[i] /= mix(1.0, determinant(affineMatrix[i]), clamp(determinant(affineMatrix[i])*3.0 - 2.0, 0.0, 1.0));
		//affineTransforms[i][0] = affineMatrix[i].getMatrix()[0];
		//affineTransforms[i][1] = affineMatrix[i].getMatrix()[4];
		//affineTransforms[i][2] = affineMatrix[i].getMatrix()[12];
		//affineTransforms[i][3] = affineMatrix[i].getMatrix()[1];
		//affineTransforms[i][4] = affineMatrix[i].getMatrix()[5];
		//affineTransforms[i][5] = affineMatrix[i].getMatrix()[13];

	}
	void(*unlinearPointer[14])(sf::Vector2f &point) = { spherical,swirl,handkerchief,sinusoidal,linear,horseshoe,polar,disc,heart,spiral,hyperbolic,diamond,ex,julia };//array of pointers to functions

	int max = 0;
	sf::Vector2f point = sf::Vector2f(0, 0);//be aware with translation 0 in matrix can cause division by 0
	float cad = 0;
	auto histogram = new float[501][501][2]();
	std::random_device rd;
	std::default_random_engine e(rd());
	std::uniform_real_distribution<> d(0.0, 1.0);

	float probability0 = affineTransforms[0][7];
	float probability1 = probability0 + affineTransforms[1][7];
	float probability2 = probability1 + affineTransforms[2][7];
	float probability3 = probability2 + affineTransforms[3][7];
	float probability4 = probability3 + affineTransforms[4][7];
	float probability5 = probability4 + affineTransforms[5][7];

	int temp, temp2;
	for (int i = 0; i < 5000000; i++)
	{

		float result = d(e);

		
		if (result < probability0) { point = affineMatrix[0] * point; cad = (cad + affineTransforms[0][6]) / 2; unlinearPointer[current[0]](point); point = postMatrix[0] * point; }
		else if (result < probability1) { point = affineMatrix[1] * point; cad = (cad + affineTransforms[1][6]) / 2; unlinearPointer[current[1]](point); point = postMatrix[1] * point;}
		else if (result <probability2) { point = affineMatrix[2] * point; cad = (cad + affineTransforms[2][6]) / 2; unlinearPointer[current[2]](point); point = postMatrix[2] * point;}
		else if (result <probability3) { point = affineMatrix[3] * point; cad = (cad + affineTransforms[3][6]) / 2; unlinearPointer[current[3]](point); point = postMatrix[3] * point;}
		else if (result <probability4) { point = affineMatrix[4] * point; cad = (cad + affineTransforms[4][6]) / 2; unlinearPointer[current[4]](point); point = postMatrix[4] * point;}
		else { point =  affineMatrix[5] * point ; cad = (cad + affineTransforms[5][6]) / 2; unlinearPointer[current[5]](point); point = postMatrix[5] * point;}

	
		if (i > 20) {
			if (abs(point.x) > 5 * zoom || abs(point.y) > 5 * zoom)
				continue;
			temp = (unsigned int)((point.x + 5 * zoom) * (50 / zoom));
			temp2 = (unsigned int)((point.y + 5 * zoom) * (50 / zoom));
			histogram[temp][temp2][0] += 1;
			if (histogram[temp][temp2][0] > max)
				max = histogram[temp][temp2][0];
			histogram[temp][temp2][1] = (histogram[temp][temp2][1] + cad) / 2;



		}

	}
	sf::Image img;
	img.create(500, 500);
	float freq = 0;
	float color = 0;
	float alpha = 0;

	for (int i = 0; i < 500; i++)
		for (int y = 0; y < 500; y++) {
			freq = histogram[y][i][0];
			color = histogram[y][i][1];
			if (freq != 0)
				alpha = color*pow((log(freq) / log(max)), 1 / 2.2);
			else
				alpha = 0;

			sf::Color col(alpha * 255, alpha * 255, alpha * 255, 255);
			img.setPixel(y, i, col);
		}
	tex.loadFromImage(img);
	screen.setTexture(tex);
	//img.saveToFile("overview.png");
	delete[] histogram;

	//}


}

int saveOffset(string name, int length, int affineNum, bool unlinearNum, float zoom, int gauss, int iterationsInMld, float(*affineTransforms)[8], int *current, nk_context *ctx) {

	string n = name.substr(0, length);
	if (length > 20) {
		return 1;
	}
	fstream k("offset.txt");
	string archive;
	stringstream buffer;
	buffer << k.rdbuf();
	archive = buffer.str();
	if (archive.find("#" + n + "\n") != string::npos) {
		return 2;
	}
	k << "\n#" << n << endl;
	k << affineNum << " " << unlinearNum << " " << zoom << " " << boundedFunctionScale << " " << gauss << " " << iterationsInMld << endl;
	for (int i = 0; i < affineNum; i++) {
		for (int y = 0; y < 8; y++)
			k << affineTransforms[i][y] << " ";
		k << endl;
	}
	for (int i = 0; i <= unlinearNum * 5; i++)
		k << current[i] << " ";
	k << endl;
	k << "$end" << endl;
	k.close();
	return 0;
}
void loadNames(string *container, int *namesNum) {
	*namesNum = 0;
	container->clear();
	fstream k("offset.txt");
	string temp;
	stringstream buffer;
	buffer << k.rdbuf();
	temp = buffer.str();
	size_t start = 0;
	size_t end = 0;
	for (int i = 0;; i++) {
		start = temp.find("#", end);
		if (start == string::npos)
			break;

		start++;
		end = temp.find("\n", start);
		*container += temp.substr(start, end - start) + '\0';
		*namesNum = i + 1;
	}

	k.close();
}
void loadOffset(string name, int currentName, int *affineNum, bool *unlinearNum, float *zoom, int *gauss, int *iterationsInMld, float(*affineTransforms)[8], int *current) {
	fstream k("offset.txt");
	string archive;
	stringstream buffer;
	buffer << k.rdbuf();
	archive = buffer.str();


	int pos = 0;
	for (int i = 0; i <= currentName; i++) {
		pos = archive.find("#", pos + 1);
	}

	pos = archive.find("\n", pos + 1);
	pos += 1;


	size_t end = archive.find(" ", pos);
	*affineNum = stoi(archive.substr(pos, end - pos));

	pos += end - pos + 1;
	end = archive.find(" ", pos);
	*unlinearNum = stoi(archive.substr(pos, end - pos));

	pos += end - pos + 1;
	end = archive.find(" ", pos);
	*zoom = stof(archive.substr(pos, end - pos));

	pos += end - pos + 1;
	end = archive.find(" ", pos);
	boundedFunctionScale = stof(archive.substr(pos, end - pos));

	pos += end - pos + 1;
	end = archive.find(" ", pos);
	*gauss = stoi(archive.substr(pos, end - pos));

	pos += end - pos + 1;
	end = archive.find("\n", pos);
	*iterationsInMld = stoi(archive.substr(pos, end - pos));

	pos += end - pos + 1;


	for (int i = 0; i < *affineNum; i++) {
		for (int y = 0; y < 8; y++) {
			end = archive.find(" ", pos);
			affineTransforms[i][y] = stof(archive.substr(pos, end - pos));
			pos += end - pos + 1;
		}
	}

	for (int i = 0; i <= *unlinearNum * 5; i++) {
		end = archive.find(" ", pos);
		current[i] = stoi(archive.substr(pos, end - pos));
		pos += end - pos + 1;
	}
	k.close();
}
void deleteOffset(int deleteNum) {

	fstream k("offset.txt");
	string archive;
	stringstream buffer;
	buffer << k.rdbuf();
	archive = buffer.str();
	k.close();
	int pos = 0;
	int end;
	for (int i = 0; i <= deleteNum; i++)
		pos = archive.find("#", pos + 1);

	pos -= 1;
	end = archive.find("$", pos);
	end += 5;
	archive.erase(pos, end - pos);
	k.open("offset.txt", ios::out | ios::trunc);
	k << archive;
	k.close();
}
int main() {
	int winH = 1080;
	//overview
	sf::View view(sf::FloatRect(0, 0, 500, 500));
	view.setViewport(sf::FloatRect(0.474, 0.2685, 0.26, 0.463));
	sf::View defaultView;


	sf::Font arial;
	arial.loadFromFile("arial.ttf");
	struct nk_user_font font;
	font.userdata.ptr = &arial;
	font.height = 15;
	font.width = width_calculation;
	struct nk_context ctx;
	nk_init_default(&ctx, &font);



	sf::ContextSettings settings;
	settings.antialiasingLevel = 8;
	sf::RenderWindow window(sf::VideoMode(1920, 1080), "IFS Flame Fractal Renderer", sf::Style::Default, settings);
	window.setFramerateLimit(60);



	//0-5 matrix affine
	//6 color
	//7 probability
	auto affineTransforms = new float[6][8]();
	auto postTransforms = new float[6][6]();
	for (int i = 0; i < 6; i++) {
		postTransforms[i][0] = 1;
		postTransforms[i][4] = 1;
	}

	int current[6] = { 0,0,0,0,0,0 };
	string names;
	int namesNum = 0;
	loadNames(&names, &namesNum);

	int affineNum = 2;
	bool unlinearNum = 0;
	int renderType = 0;
	float zoom = 1;
	int gauss = 9;
	int iterationsInMld = 9;

	int textLen = 0;
	int popup = 0;
	int currentName = 0;
	while (window.isOpen()) {

		//INPUT
		sf::Event event;
		nk_input_begin(&ctx);
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();

			if (event.type == sf::Event::Resized)
			{
				winH = event.size.height;
				// update the view to the new size of the window
				sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
				window.setView(sf::View(visibleArea));
			}
			eventsToGui(&event, &ctx);
		}
		nk_input_end(&ctx);


		//GUI
		enum { EASY, HARD };
		static int op = EASY;
		static float value = 0.6f;
		static int i = 20;
		static char text[30];
		if (nk_begin(&ctx, "LeftPanel", nk_rect(0, 0, 400, winH),
			NK_WINDOW_BORDER)) {


			nk_layout_row_static(&ctx, 30, 360, 1);
			nk_label(&ctx, "IFS FLAME FRACTAL RENDERER", NK_TEXT_CENTERED);


			nk_layout_row_begin(&ctx, NK_STATIC, 40, 2);
			{
				nk_layout_row_push(&ctx, 170);
				nk_label(&ctx, "Affine transforms:", NK_TEXT_LEFT);
				nk_layout_row_push(&ctx, 80);
				nk_property_int(&ctx, "#", 2, &affineNum, 6, 1, 0);
			}
			nk_layout_row_end(&ctx);

			nk_layout_row_begin(&ctx, NK_STATIC, 40, 2);
			{
				nk_layout_row_push(&ctx, 170);
				nk_label(&ctx, "Zoom:", NK_TEXT_LEFT);
				nk_layout_row_push(&ctx, 80);
				nk_property_float(&ctx, "#", 0.1, &zoom, 10, 0.1, 0);
			}
			nk_layout_row_end(&ctx);

			nk_layout_row_begin(&ctx, NK_STATIC, 40, 2);
			{
				nk_layout_row_push(&ctx, 170);
				nk_label(&ctx, "Bounded function scale:", NK_TEXT_LEFT);
				nk_layout_row_push(&ctx, 80);
				nk_property_float(&ctx, "#", 1, &boundedFunctionScale, 100, 1, 0);
			}
			nk_layout_row_end(&ctx);

			nk_layout_row_begin(&ctx, NK_STATIC, 40, 2);
			{
				nk_layout_row_push(&ctx, 170);
				nk_label(&ctx, "Gaussian blur:", NK_TEXT_LEFT);
				nk_layout_row_push(&ctx, 80);
				nk_property_int(&ctx, "#", 0, &gauss, 10, 1, 0);
			}
			nk_layout_row_end(&ctx);

			nk_layout_row_begin(&ctx, NK_STATIC, 40, 2);
			{
				nk_layout_row_push(&ctx, 170);
				nk_label(&ctx, "Iterations in billions:", NK_TEXT_LEFT);
				nk_layout_row_push(&ctx, 80);
				nk_property_int(&ctx, "#", 1, &iterationsInMld, 100, 1, 0);
			}
			nk_layout_row_end(&ctx);

			nk_layout_row_static(&ctx, 20, 300, 0);
			nk_layout_row_static(&ctx, 30, 300, 1);
			nk_label(&ctx, "Unlinear transforms:", NK_TEXT_LEFT);
			nk_layout_row_static(&ctx, 30, 200, 1);
			if (nk_option_label(&ctx, "one for all affine", unlinearNum == 0)) unlinearNum = 0;
			nk_layout_row_static(&ctx, 30, 200, 1);
			if (nk_option_label(&ctx, "one for each affine", unlinearNum == 1)) unlinearNum = 1;

			nk_layout_row_static(&ctx, 20, 300, 0);
			nk_layout_row_static(&ctx, 30, 300, 1);
			nk_label(&ctx, "Render type:", NK_TEXT_LEFT);
			nk_layout_row_static(&ctx, 30, 200, 1);
			if (nk_option_label(&ctx, "CPU 1 thread", renderType == 0)) renderType = 0;
			nk_layout_row_static(&ctx, 30, 200, 1);
			if (nk_option_label(&ctx, "CPU multithread", renderType == 1)) renderType = 1;
			nk_layout_row_static(&ctx, 30, 200, 1);
			if (nk_option_label(&ctx, "GPU CUDA", renderType == 2)) renderType = 2;


			nk_layout_row_static(&ctx, 30, 300, 1);
			if (nk_button_label(&ctx, "Render low quality to overview"))
				renderToOverview(renderType, affineTransforms, current, unlinearNum, affineNum, window, zoom,postTransforms);
			nk_layout_row_static(&ctx, 10, 300, 0);
			nk_layout_row_static(&ctx, 30, 300, 1);
			if (nk_button_label(&ctx, "Render high quality to file"))
				renderToFile(renderType, affineTransforms, current, unlinearNum, affineNum, zoom, gauss, iterationsInMld,postTransforms);
			nk_layout_row_static(&ctx, 30, 300, 0);

			if (nk_tree_push(&ctx, NK_TREE_TAB, "Offsets", NK_MINIMIZED)) {
				nk_layout_row_dynamic(&ctx, 30, 3);
				nk_label(&ctx, "Name:", NK_TEXT_LEFT);
				nk_edit_string(&ctx, NK_EDIT_SIMPLE, text, &textLen, 30, nk_filter_default);
				if (nk_button_label(&ctx, "Save")) {
					popup = saveOffset(text, textLen, affineNum, unlinearNum, zoom, gauss, iterationsInMld, affineTransforms, current, &ctx);
					loadNames(&names, &namesNum);

				}
				if (popup == 2) {
					static struct nk_rect s = { 500, 540, 220, 90 };
					if (nk_popup_begin(&ctx, NK_POPUP_STATIC, "Critical error", 0, s))
					{
						nk_layout_row_dynamic(&ctx, 30, 1);
						nk_label(&ctx, "Name already exists", NK_TEXT_LEFT);
						nk_layout_row_dynamic(&ctx, 30, 1);
						if (nk_button_label(&ctx, "OK")) {
							popup = 0;
							nk_popup_close(&ctx);
						}
						nk_popup_end(&ctx);
					}
				}
				else if (popup == 1) {
					static struct nk_rect s = { 500, 540, 220, 90 };
					if (nk_popup_begin(&ctx, NK_POPUP_STATIC, "Critical error", 0, s))
					{
						nk_layout_row_dynamic(&ctx, 30, 1);
						nk_label(&ctx, "Name is too long", NK_TEXT_LEFT);
						nk_layout_row_dynamic(&ctx, 30, 1);
						if (nk_button_label(&ctx, "OK")) {
							popup = 0;
							nk_popup_close(&ctx);
						}
						nk_popup_end(&ctx);
					}
				}
				else if (popup == 3) {
					static struct nk_rect s = { 500, 540, 220, 90 };
					if (nk_popup_begin(&ctx, NK_POPUP_STATIC, "Confirm deletion", 0, s))
					{
						nk_layout_row_dynamic(&ctx, 30, 1);
						nk_label(&ctx, "Delete is permanently", NK_TEXT_LEFT);
						nk_layout_row_dynamic(&ctx, 30, 2);
						if (nk_button_label(&ctx, "OK")) {
							deleteOffset(currentName);
							currentName = 0;
							loadNames(&names, &namesNum);
							popup = 0;
							nk_popup_close(&ctx);
						}
						if (nk_button_label(&ctx, "Cancel")) {
							popup = 0;
							nk_popup_close(&ctx);
						}
						nk_popup_end(&ctx);
					}

				}



				nk_layout_row_dynamic(&ctx, 30, 3);
				currentName = nk_combo_string(&ctx, &names[0], currentName, namesNum, 30, nk_vec2(200, 200));
				if (nk_button_label(&ctx, "Delete")) {
					popup = 3;
				}
				if (nk_button_label(&ctx, "Load")) {
					loadOffset(names, currentName, &affineNum, &unlinearNum, &zoom, &gauss, &iterationsInMld, affineTransforms, current);
				}

				nk_tree_pop(&ctx);
			}


			if (nk_tree_push(&ctx, NK_TREE_TAB, "Affine transforms", NK_MINIMIZED)) {
				static float det=0;
				for (int i = 0; i < affineNum; i++) {
					nk_layout_row_static(&ctx, 30, 300, 1);
					nk_label(&ctx, "Transformation matrix:", NK_TEXT_LEFT);
					nk_layout_row_dynamic(&ctx, 30, 3);
					nk_property_float(&ctx, "#", INT_MIN, &affineTransforms[i][0], INT_MAX, 0.1, 0);
					nk_property_float(&ctx, "#", INT_MIN, &affineTransforms[i][1], INT_MAX, 0.1, 0);
					nk_property_float(&ctx, "#", INT_MIN, &affineTransforms[i][2], INT_MAX, 0.1, 0);
					nk_layout_row_dynamic(&ctx, 30, 3);
					nk_property_float(&ctx, "#", INT_MIN, &affineTransforms[i][3], INT_MAX, 0.1, 0);
					nk_property_float(&ctx, "#", INT_MIN, &affineTransforms[i][4], INT_MAX, 0.1, 0);
					nk_property_float(&ctx, "#", INT_MIN, &affineTransforms[i][5], INT_MAX, 0.1, 0);
					nk_layout_row_dynamic(&ctx, 10, 0);
					nk_layout_row_dynamic(&ctx, 30, 1);
					nk_property_float(&ctx, "#Color grayscale", 0, &affineTransforms[i][6], 1, 0.1, 0);
					nk_layout_row_dynamic(&ctx, 30, 1);
					nk_property_float(&ctx, "#Probability", 0, &affineTransforms[i][7], 1, 0.1, 0);
					nk_layout_row_dynamic(&ctx, 10, 0);
					nk_layout_row_static(&ctx, 30, 100, 3);
					if (nk_button_label(&ctx, "Clear")) {
						for (int y = 0; y < 8; y++)
							affineTransforms[i][y] = 0;
					}
					det = affineTransforms[i][0] * affineTransforms[i][4] - affineTransforms[i][1] * affineTransforms[i][3];
					nk_label(&ctx, "  Determinant:", NK_TEXT_LEFT);
					nk_label(&ctx, to_string(det).c_str(), NK_TEXT_LEFT);
					if (nk_tree_push_id(&ctx, NK_TREE_NODE, "Imminent transforms", NK_MINIMIZED,i)) { 
						static auto *temp=new float[6][8]();
						nk_layout_row_dynamic(&ctx, 30, 4);
						nk_label(&ctx, "Translation:", NK_TEXT_LEFT);
						nk_property_float(&ctx, "#x:", INT_MIN, &temp[i][0], INT_MAX, 0.1, 0.1);
						nk_property_float(&ctx, "#y:", INT_MIN, &temp[i][1], INT_MAX, 0.1, 0.1);
						if (nk_button_label(&ctx, "Apply")) {
							affineTransforms[i][2] = affineTransforms[i][0] * temp[i][0] + affineTransforms[i][1] * temp[i][1] + affineTransforms[i][2];
							affineTransforms[i][5] = affineTransforms[i][3] * temp[i][0] + affineTransforms[i][4] * temp[i][1] + affineTransforms[i][5];
						}
						nk_layout_row_dynamic(&ctx, 30, 4);
						nk_label(&ctx, "Scale:", NK_TEXT_LEFT);
						nk_property_float(&ctx, "#x:", INT_MIN, &temp[i][2], INT_MAX, 0.1, 0.1);
						nk_property_float(&ctx, "#y:", INT_MIN, &temp[i][3], INT_MAX, 0.1, 0.1);
						if (nk_button_label(&ctx, "Apply")) {
							affineTransforms[i][0] = affineTransforms[i][0] * temp[i][2];
							affineTransforms[i][1] = affineTransforms[i][1] * temp[i][3];

							affineTransforms[i][3] = affineTransforms[i][3] * temp[i][2];
							affineTransforms[i][4] = affineTransforms[i][4] * temp[i][3];
						}
						nk_layout_row_dynamic(&ctx, 30, 4);
						nk_label(&ctx, "Shear:", NK_TEXT_LEFT);
						nk_property_float(&ctx, "#x:", INT_MIN, &temp[i][4], INT_MAX, 0.1, 0.1);
						nk_property_float(&ctx, "#y:", INT_MIN, &temp[i][5], INT_MAX, 0.1, 0.1);
						if (nk_button_label(&ctx, "Apply")) {
							affineTransforms[i][0] = affineTransforms[i][0]+ affineTransforms[i][1]*temp[i][5];
							affineTransforms[i][1] = affineTransforms[i][0] * temp[i][4]+ affineTransforms[i][1];

							affineTransforms[i][3] = affineTransforms[i][3] + affineTransforms[i][4]* temp[i][5];
							affineTransforms[i][4] = affineTransforms[i][3] * temp[i][4]+ affineTransforms[i][4];
						}
						nk_layout_row_begin(&ctx, NK_STATIC, 30, 3);
						nk_layout_row_push(&ctx, 82);
						nk_label(&ctx, "Rotate:", NK_TEXT_LEFT);
						nk_layout_row_push(&ctx, 169);
						nk_property_float(&ctx, "#Angle:", INT_MIN, &temp[i][6], INT_MAX, 0.1, 0.1);
						nk_layout_row_push(&ctx, 83);
						if (nk_button_label(&ctx, "Apply")) {
							temp[i][7] = temp[i][6] * 3.141592/180;
							affineTransforms[i][0] = affineTransforms[i][0] * cos(temp[i][7]) + affineTransforms[i][1] * sin(temp[i][7]);
							affineTransforms[i][1] = -affineTransforms[i][0] * sin(temp[i][7]) + affineTransforms[i][1] * cos(temp[i][7]);

							affineTransforms[i][3] = affineTransforms[i][3] * cos(temp[i][7]) + affineTransforms[i][4] * sin(temp[i][7]);
							affineTransforms[i][4] = -affineTransforms[i][3] * sin(temp[i][7]) + affineTransforms[i][4] * cos(temp[i][7]);
						}

						
						
						
						
						nk_tree_pop(&ctx); }
					if (nk_tree_push_id(&ctx, NK_TREE_NODE, "Post Transform", NK_MINIMIZED, i)) {
						nk_layout_row_dynamic(&ctx, 30, 3);
						nk_property_float(&ctx, "#", INT_MIN, &postTransforms[i][0], INT_MAX, 0.1, 0);
						nk_property_float(&ctx, "#", INT_MIN, &postTransforms[i][1], INT_MAX, 0.1, 0);
						nk_property_float(&ctx, "#", INT_MIN, &postTransforms[i][2], INT_MAX, 0.1, 0);
						nk_layout_row_dynamic(&ctx, 30, 3);
						nk_property_float(&ctx, "#", INT_MIN, &postTransforms[i][3], INT_MAX, 0.1, 0);
						nk_property_float(&ctx, "#", INT_MIN, &postTransforms[i][4], INT_MAX, 0.1, 0);
						nk_property_float(&ctx, "#", INT_MIN, &postTransforms[i][5], INT_MAX, 0.1, 0);
						nk_layout_row_dynamic(&ctx, 10, 0);
						nk_layout_row_static(&ctx, 30, 100, 3);
						if (nk_button_label(&ctx, "Clear")) {
							for (int y = 0; y < 6; y++)
								postTransforms[i][y] = 0;
						}
						det = postTransforms[i][0] * postTransforms[i][4] - postTransforms[i][1] * postTransforms[i][3];
						nk_label(&ctx, "  Determinant:", NK_TEXT_LEFT);
						nk_label(&ctx, to_string(det).c_str(), NK_TEXT_LEFT);
						if (nk_tree_push_id(&ctx, NK_TREE_NODE, "Imminent post transforms", NK_MINIMIZED, i)) {
							static auto *temp = new float[6][8]();
							nk_layout_row_dynamic(&ctx, 30, 4);
							nk_label(&ctx, "Translation:", NK_TEXT_LEFT);
							nk_property_float(&ctx, "#x:", INT_MIN, &temp[i][0], INT_MAX, 0.1, 0.1);
							nk_property_float(&ctx, "#y:", INT_MIN, &temp[i][1], INT_MAX, 0.1, 0.1);
							if (nk_button_label(&ctx, "Apply")) {
								postTransforms[i][2] = postTransforms[i][0] * temp[i][0] + postTransforms[i][1] * temp[i][1] + postTransforms[i][2];
								postTransforms[i][5] = postTransforms[i][3] * temp[i][0] + postTransforms[i][4] * temp[i][1] + postTransforms[i][5];
							}
							nk_layout_row_dynamic(&ctx, 30, 4);
							nk_label(&ctx, "Scale:", NK_TEXT_LEFT);
							nk_property_float(&ctx, "#x:", INT_MIN, &temp[i][2], INT_MAX, 0.1, 0.1);
							nk_property_float(&ctx, "#y:", INT_MIN, &temp[i][3], INT_MAX, 0.1, 0.1);
							if (nk_button_label(&ctx, "Apply")) {
								postTransforms[i][0] = postTransforms[i][0] * temp[i][2];
								postTransforms[i][1] = postTransforms[i][1] * temp[i][3];

								postTransforms[i][3] = postTransforms[i][3] * temp[i][2];
								postTransforms[i][4] = postTransforms[i][4] * temp[i][3];
							}
							nk_layout_row_dynamic(&ctx, 30, 4);
							nk_label(&ctx, "Shear:", NK_TEXT_LEFT);
							nk_property_float(&ctx, "#x:", INT_MIN, &temp[i][4], INT_MAX, 0.1, 0.1);
							nk_property_float(&ctx, "#y:", INT_MIN, &temp[i][5], INT_MAX, 0.1, 0.1);
							if (nk_button_label(&ctx, "Apply")) {
								postTransforms[i][0] = postTransforms[i][0] + postTransforms[i][1] * temp[i][5];
								postTransforms[i][1] = postTransforms[i][0] * temp[i][4] + postTransforms[i][1];

								postTransforms[i][3] = postTransforms[i][3] + postTransforms[i][4] * temp[i][5];
								postTransforms[i][4] = postTransforms[i][3] * temp[i][4] + postTransforms[i][4];
							}
							nk_layout_row_begin(&ctx, NK_STATIC, 30, 3);
							nk_layout_row_push(&ctx, 79);
							nk_label(&ctx, "Rotate:", NK_TEXT_LEFT);
							nk_layout_row_push(&ctx, 162);
							nk_property_float(&ctx, "#Angle:", INT_MIN, &temp[i][6], INT_MAX, 0.1, 0.1);
							nk_layout_row_push(&ctx, 79);
							if (nk_button_label(&ctx, "Apply")) {
								temp[i][7] = temp[i][6] * 3.141592 / 180;
								postTransforms[i][0] = postTransforms[i][0] * cos(temp[i][7]) + postTransforms[i][1] * sin(temp[i][7]);
								postTransforms[i][1] = -postTransforms[i][0] * sin(temp[i][7]) + postTransforms[i][1] * cos(temp[i][7]);

								postTransforms[i][3] = postTransforms[i][3] * cos(temp[i][7]) + postTransforms[i][4] * sin(temp[i][7]);
								postTransforms[i][4] = -postTransforms[i][3] * sin(temp[i][7]) + postTransforms[i][4] * cos(temp[i][7]);
							}



					nk_tree_pop(&ctx);
					}
					nk_tree_pop(&ctx);
					}
					nk_layout_row_dynamic(&ctx, 30, 0);

					





				}
				nk_tree_pop(&ctx);
			}
			if (nk_tree_push(&ctx, NK_TREE_TAB, "Unlinear transforms", NK_MINIMIZED)) {
				static const char *functions[] = { "Spherical","Swirl","Handkerchief","Sinusoidal","Linear","Horseshoe","Polar","Disc","Heart","Spiral","Hyperbolic","Diamond","Ex","Julia" };
				if (unlinearNum == 0) {
					nk_layout_row_static(&ctx, 25, 200, 1);
					current[0] = nk_combo(&ctx, functions, sizeof(functions) / sizeof(functions)[0], current[0], 25, nk_vec2(200, 200));

				}
				else for (int i = 0; i < affineNum; i++) {
					nk_layout_row_static(&ctx, 25, 200, 1);
					current[i] = nk_combo(&ctx, functions, sizeof(functions) / sizeof(functions)[0], current[i], 25, nk_vec2(200, 200));
					nk_layout_row_static(&ctx, 10, 200, 0);
				}

				nk_tree_pop(&ctx);
			}


			nk_layout_row_static(&ctx, 300, 300, 0);




		}
		nk_end(&ctx);

		window.clear();

		//DRAW
		const struct nk_command *cmd = 0;
		nk_foreach(cmd, &ctx) {
			switch (cmd->type) {
			case NK_COMMAND_RECT_FILLED:
				nuklearSfmlDrawRectFilled(cmd, window);
				break;
			case NK_COMMAND_TEXT:
				nuklearSfmlDrawText(cmd, window);
				break;
			case NK_COMMAND_SCISSOR:
				nuklearSfmlDrawScissor(cmd, window);
				break;
			case NK_COMMAND_RECT:
				nuklearSfmlDrawRectOutline(cmd, window);
				break;
			case NK_COMMAND_CIRCLE_FILLED:
				nuklearSfmlDrawCircleFilled(cmd, window);
				break;
			case NK_COMMAND_TRIANGLE_FILLED:
				nuklearSfmlDrawTriangleFilled(cmd, window);
				break;

			}

		}
		nk_clear(&ctx);

		defaultView = window.getView();
		window.setView(view);
		window.draw(screen);
		window.setView(defaultView);
		window.display();
	}


	return 0;
}


