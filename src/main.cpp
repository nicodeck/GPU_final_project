#include "../kernels/main.cuh"
#include "cam_params.hpp"
#include "constants.hpp"
#include "graph.h"

#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>

#define SHRT_MAX 32767

std::vector<cam> read_cams(std::string const& folder)
{
	// Init parameters
	std::vector<params<double>> cam_params_vector = get_cam_params();

	// Init cameras
	std::vector<cam> cam_array(cam_params_vector.size());
	for (int i = 0; i < cam_params_vector.size(); i++)
	{
		// Name
		std::string name = folder + "/v" + std::to_string(i) + ".png";

		// Read PNG file
		cv::Mat im_rgb = cv::imread(name);
		cv::Mat im_yuv;
		const int width = im_rgb.cols;
		const int height = im_rgb.rows;

		// Convert to YUV420
		cv::cvtColor(im_rgb, im_yuv, cv::COLOR_BGR2YUV_I420);
		const int size = width * height * 1.5; // YUV 420

		std::vector<cv::Mat> YUV;
		cv::split(im_rgb, YUV);

		// Params
		cam_array.at(i) = cam(name, width, height, size, YUV, cam_params_vector.at(i));
	}

	return cam_array;

	// cv::Mat U(height / 2, width / 2, CV_8UC1, cam_array.at(0).image.data() + (int)(width * height * 1.25));
	// cv::namedWindow("im", cv::WINDOW_NORMAL);
	// cv::imshow("im", U);
	// cv::waitKey(0);
}

std::vector<cv::Mat> sweeping_plane(cam const ref, std::vector<cam> const& cam_vector, int window = 3)
{
	// Initialization to MAX value
	// std::vector<float> cost_cube(ref.width * ref.height * ZPlanes, 255.f);
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	printf("%d * %d", ref.width, ref.height);

	// For each camera in the setup (reference is skipped)
	for (auto& cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		std::cout << "Cam: " << cam.name << std::endl;
		// For each pixel and candidate: (i) calculate projection index, (ii) calculate cost against reference, (iii) store minimum cost
		for (int zi = 0; zi < ZPlanes; zi++)
		{
			std::cout << "Plane " << zi << std::endl;
			for (int y = 0; y < ref.height; y++)
			{
				for (int x = 0; x < ref.width; x++)
				{
					// (i) calculate projection index

					// Calculate z from ZNear, ZFar and ZPlanes (projective transformation) (zi = 0, z = ZFar)
					double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

					// 2D ref camera point to 3D in ref camera coordinates (p * K_inv)
					double X_ref = (ref.p.K_inv[0] * x + ref.p.K_inv[1] * y + ref.p.K_inv[2]) * z;
					double Y_ref = (ref.p.K_inv[3] * x + ref.p.K_inv[4] * y + ref.p.K_inv[5]) * z;
					double Z_ref = (ref.p.K_inv[6] * x + ref.p.K_inv[7] * y + ref.p.K_inv[8]) * z;

					// 3D in ref camera coordinates to 3D world
					double X = ref.p.R_inv[0] * X_ref + ref.p.R_inv[1] * Y_ref + ref.p.R_inv[2] * Z_ref - ref.p.t_inv[0];
					double Y = ref.p.R_inv[3] * X_ref + ref.p.R_inv[4] * Y_ref + ref.p.R_inv[5] * Z_ref - ref.p.t_inv[1];
					double Z = ref.p.R_inv[6] * X_ref + ref.p.R_inv[7] * Y_ref + ref.p.R_inv[8] * Z_ref - ref.p.t_inv[2];

					// 3D world to projected camera 3D coordinates
					double X_proj = cam.p.R[0] * X + cam.p.R[1] * Y + cam.p.R[2] * Z - cam.p.t[0];
					double Y_proj = cam.p.R[3] * X + cam.p.R[4] * Y + cam.p.R[5] * Z - cam.p.t[1];
					double Z_proj = cam.p.R[6] * X + cam.p.R[7] * Y + cam.p.R[8] * Z - cam.p.t[2];

					// Projected camera 3D coordinates to projected camera 2D coordinates
					double x_proj = (cam.p.K[0] * X_proj / Z_proj + cam.p.K[1] * Y_proj / Z_proj + cam.p.K[2]);
					double y_proj = (cam.p.K[3] * X_proj / Z_proj + cam.p.K[4] * Y_proj / Z_proj + cam.p.K[5]);
					double z_proj = Z_proj;

					x_proj = x_proj < 0 || x_proj >= cam.width ? 0 : roundf(x_proj);
					y_proj = y_proj < 0 || y_proj >= cam.height ? 0 : roundf(y_proj);

					// (ii) calculate cost against reference
					// Calculating cost in a window
					float cost = 0.0f;
					float cc = 0.0f;
					for (int k = -window / 2; k <= window / 2; k++)
					{
						for (int l = -window / 2; l <= window / 2; l++)
						{
							if (x + l < 0 || x + l >= ref.width)
								continue;
							if (y + k < 0 || y + k >= ref.height)
								continue;
							if (x_proj + l < 0 || x_proj + l >= cam.width)
								continue;
							if (y_proj + k < 0 || y_proj + k >= cam.height)
								continue;

							// Y
							cost += fabs(ref.YUV[0].at<uint8_t>(y + k, x + l) - cam.YUV[0].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
							// U
							// cost += fabs(ref.YUV[1].at<uint8_t >(y + k, x + l) - cam.YUV[1].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
							// V
							// cost += fabs(ref.YUV[2].at<uint8_t >(y + k, x + l) - cam.YUV[2].at<uint8_t>((int)y_proj + k, (int)x_proj + l));
							cc += 1.0f;
						}
					}
					cost /= cc;

					//  (iii) store minimum cost (arranged as cost images, e.g., first image = cost of every pixel for the first candidate)
					// only the minimum cost for all the cameras is stored
					cost_cube[zi].at<float>(y, x) = fminf(cost_cube[zi].at<float>(y, x), cost);
				}
			}
		}
	}

	// Visualize costs
	// for (int zi = 0; zi < ZPlanes; zi++)
	// {
	// 	std::cout << "plane " << zi << std::endl;
	// 	cv::namedWindow("Cost", cv::WINDOW_NORMAL);
	// 	cv::imshow("Cost", cost_cube.at(zi) / 255.f);
	// 	cv::waitKey(0);
	// }
	return cost_cube;
}

cv::Mat find_min(std::vector<cv::Mat> const& cost_cube)
{
	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	cv::Mat ret(height, width, CV_32FC1, 255.);
	cv::Mat depth(height, width, CV_8U, 255);

	for (int zi = 0; zi < zPlanes; zi++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (cost_cube[zi].at<float>(y, x) < ret.at<float>(y, x))
				{
					ret.at<float>(y, x) = cost_cube[zi].at<float>(y, x);
					depth.at<u_char>(y, x) = zi;
				}
			}
		}
	}

	return depth;
}

/*The next two function are used to perform the grpah cut on the results
DO NOT MODIFY THOSE FUNCTIONS - DO NOT TRY TO IMPLEMENT THEM ON THE GPU*/
void depth_estimation_by_graph_cut_sWeight_add_nodes(Graph& g, std::vector<Graph::node_id>& nodes, cv::Size destPixel, cv::Size sourcePixel, cv::Size imgSize, std::vector<double> m_aiEdgeCost, cv::Mat1w labels, int label, double cost_cur) {
	const int idxSourcePixel = sourcePixel.height * imgSize.width + sourcePixel.width;
	const int idxDestPixel = destPixel.height * imgSize.width + destPixel.width;
	const double cost_cur_temp = cost_cur;

	if (labels(sourcePixel.height, sourcePixel.width) != labels(destPixel.height, destPixel.width)) {
		//add a new node and add edge between it and the adjacent nodes
		Graph::node_id tmp_node = g.add_node();
		const double cost_temp = m_aiEdgeCost[std::abs(labels(destPixel.height, destPixel.width) - label)];
		g.set_tweights(tmp_node, 0, m_aiEdgeCost[std::abs(labels(sourcePixel.height, sourcePixel.width) - labels(destPixel.height, destPixel.width))]);
		g.add_edge(nodes[idxSourcePixel], tmp_node, cost_cur_temp, cost_cur_temp);
		g.add_edge(tmp_node, nodes[idxDestPixel], cost_temp, cost_temp);
	}
	else //only add an edge between two nodes
		g.add_edge(nodes[idxSourcePixel], nodes[idxDestPixel], cost_cur_temp, cost_cur_temp);
}

cv::Mat depth_estimation_by_graph_cut_sWeight(std::vector<cv::Mat> const& cost_cube) {
	//DO NOT TRY TO IMPLEMENT THIS FUNCTION ON THE GPU

	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	//To store the depth values assigned to each pixels, start with 0
	cv::Mat1w labels = cv::Mat::zeros(height, width, CV_16U);
	//store the cost for a label
	std::vector<double> m_aiEdgeCost;
	double smoothing_lambda = 1.0;
	m_aiEdgeCost.resize(zPlanes);
	for (int i = 0; i < zPlanes; ++i)
		m_aiEdgeCost[i] = smoothing_lambda * i;

	for (int source = 0; source < zPlanes; ++source) {
		printf("depth layer %i \n", source);
		Graph g;
		std::vector<Graph::node_id> nodes(height * width, nullptr);

		//Putting the weights for the connection to the source and the sink for each nodes
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				//indice global du pixel
				const int pp = r * width + c;
				nodes[pp] = g.add_node();
				const ushort label = labels(r, c);
				if (label == source)
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), SHRT_MAX);
				else
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), cost_cube[label].at<float>(r, c));
			}
		}


		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				const double cost_curr = m_aiEdgeCost[std::abs(labels(j, i) - source)];

				//create an edge between the adjacent nodes, may add an additional node on this edge if the previously calculated labels are different
				if (i != width - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i + 1, j), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
				if (j != height - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i, j + 1), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
			}
		}
		//printf("nodes and egde set \n");

		//resolve the maximum flow/minimum cut problem
		g.maxflow();

		//update the depth labels, nodes that are still connected to the source will receive a new depth label
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				const int pp = r * width + c;
				if (g.what_segment(nodes[pp]) != Graph::SOURCE)
					labels(r, c) = ushort(source);
			}
		}
		nodes.clear();

		/*
		cv::namedWindow("labels", cv::WINDOW_NORMAL);
		cv::imshow("labels", labels);
		cv::waitKey(0);
		*/

	}

	cv::Mat depth;
	labels.convertTo(depth, CV_8U, 1.0);

	return depth;
}

// PLANE SWEEPING FOR THE CUDA VERSION
std::vector<cv::Mat> sweeping_plane_cuda(int ref_index, std::vector<cam> const& cam_vector, int window = 3) {

	//Conversion of parameters into something usable by the kernel
	//Transformation of vectors and matrices into arrays

	std::cout << "Sweeping plane cuda version" << std::endl;

	//parameter of the camera
	auto& ref = cam_vector.at(0);
	int cam_width = ref.width;
	int cam_height = ref.height;
	int cam_number = cam_vector.size();

	//memory allocation to store camera settings
	float* cam_h_K = new float[cam_number * 9]; //calibration matrix (intrinsic properties of the camera)
	float* ref_h_K_inv = new float[9];
	float* cam_h_R = new float[cam_number * 9]; //rotation matrix (camera rotation)
	float* ref_h_R_inv = new float[9];
	double* cam_h_t = new double[cam_number * 3]; //translation vector (camera translation)
	double* ref_h_t_inv = new double[3];

	int* h_YUV = new int[cam_number * cam_width * cam_height];
	int* h_dim = new int[cam_number * 2];

	for (int i = 0; i < cam_number; i++) {
		auto& cam = cam_vector[i];

		h_dim[PARAM_ABSOLUTE_INDEX(i, 0, 2)] = cam.width;
		h_dim[PARAM_ABSOLUTE_INDEX(i, 1, 2)] = cam.height;

		for (int j = 0; j < 9; j++) {

			if (i == ref_index) {
				ref_h_K_inv[j] = ref.p.K_inv[j];
				ref_h_R_inv[j] = ref.p.R_inv[j];
			}
			else {
				cam_h_K[PARAM_ABSOLUTE_INDEX(i, j, 9)] = cam.p.K[j];
				cam_h_R[PARAM_ABSOLUTE_INDEX(i, j, 9)] = cam.p.R[j];
			}

			if (j < 3) {
				if (i == ref_index) {
					ref_h_t_inv[j] = ref.p.t_inv[j];
				}
				else {
					cam_h_t[PARAM_ABSOLUTE_INDEX(i, j, 3)] = cam.p.t[j];
				}
			}
		}

		for (int y = 0; y < cam_height && y < cam.height; y++) {
			for (int x = 0; x < cam_width && x < cam.width; x++) {
				h_YUV[CUBE(x, y, i, cam_width, cam_height)] = cam.YUV[0].at<uint8_t>(y, x); // Actually just luminance Y
			}
		}
	}

	float* result = new float[cam_width * cam_height * ZPlanes];


	wrap_plane_sweeping(result, cam_h_K, ref_h_K_inv, cam_h_R, ref_h_R_inv, cam_h_t, ref_h_t_inv, h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);

	/*The following lines of code reconvert the CUDA results
	(stored in 'result') into matrix vectors to be able to manipulate them
	more easily in C++ code.*/

	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int zi = 0; zi < cost_cube.size(); zi++)
	{
		cost_cube[zi] = cv::Mat(cam_height, cam_width, CV_32FC1, &result[zi * cam_width * cam_height]);
	}

	delete[] cam_h_K, ref_h_K_inv, cam_h_R, ref_h_R_inv, cam_h_t, ref_h_t_inv, h_dim; // result is not deleted as the matrices point to parts of it (they do not copy the data)
	return cost_cube;

}

int main()
{
	// Read cams
	std::vector<cam> cam_vector = read_cams("data");

	// Test call a CUDA function
	

	// Sweeping algorithm for camera 0
	auto start = start_cpu_timer();
	std::vector<cv::Mat> cost_cube = sweeping_plane(cam_vector.at(0), cam_vector, 5);
	double plane_sweep_time = end_cpu_timer(start);

	std::cout << "CPU time: " << plane_sweep_time << " ms" << std::endl;
	//std::vector<cv::Mat> cost_cube = sweeping_plane_cuda(0, cam_vector, 5);

	// Use graph cut to generate depth map 
	// Cleaner results, long compute time
	//cv::Mat depth = depth_estimation_by_graph_cut_sWeight(cost_cube);

	// Find min cost and generate depth map
	// Faster result, low quality

	cv::Mat depth = find_min(cost_cube);

	cv::namedWindow("Depth", cv::WINDOW_NORMAL);
	cv::imshow("Depth", depth);
	cv::waitKey(0);

	cv::imwrite("./depth_map.png", depth);
	

	//printf("%f", depth.at<uchar>(0, 0));

	return 0;
}