// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
// Include GLEW
#include <GL/glew.h>
//#include <Windows.h>
// Include GLFW
#include <glfw3.h>
GLFWwindow* window;
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/usc.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/apps/render_views_tesselated_sphere.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <time.h> 
#include <pcl/recognition/hv/hv_go.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
// Include GLM
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;
#include <common\shader.hpp>
#include <pcl/features/moment_of_inertia_estimation.h>

#define _USE_MATH_DEFINES
bool visualizate=true;
float fov = 57.0f;
float rad = 2.20f;
int width= 150;
int height= 150;
int merid=10;
int para=10;
std::string model_filename_;
std::string dir;

void
	showHelp (char *filename)
{
	std::cout << std::endl;
	std::cout << "***************************************************************************" << std::endl;
	std::cout << "*                                                                         *" << std::endl;
	std::cout << "*               Cad views with color  - Usage Guide                       *" << std::endl;
	std::cout << "*                                                                         *" << std::endl;
	std::cout << "***************************************************************************" << std::endl << std::endl;
	std::cout << "Usage: " << filename << " model_filename.ply" << dir <<"directory"<< std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "     -h:                  Show this help." << std::endl;
	std::cout << "     --width val:         Width of the virtual screen of the camera" << std::endl;
	std::cout << "     --height val:        Height of the virtual screen of the camera" << std::endl;
	std::cout << "     --radius val:        Radius of the sphere around the object(normalized)" << std::endl;
	std::cout << "     --fov val:			Field of view of the camera" << std::endl;
	std::cout << "     --visualizate val:   bool to choose if you want to visualize the process" << std::endl;
	std::cout << "     --meridian val:      Number of meridians" << std::endl << std::endl;
	std::cout << "     --paralel val:       Number of paralels" << std::endl << std::endl;
	std::cout << "The number of views depends on the number of meridian an paralel, taking" << std::endl ;
	std::cout << "consideration that in the 2 polesonly existe one view" << std::endl ;
	std::cout << "for example for 10 meridian and 10 paralel, the number" << std::endl ;
	std::cout << "of views is 82( 8*10 paralels + 2 poles)" << std::endl ;
}
void calculatecampoints(std::vector<glm::vec3>& pos,float radius, int meridian,int paralel)
{
	float const R = 1./(float)(paralel-1);
     float const S = 1./(float)(meridian-1);

	// Calc The Vertices
for (int i = 0; i < paralel; ++i){

    // Loop Through Slices
    for (int j = 0; j <  meridian; ++j){

        // Calc The Vertex Positions
        float x = cos(2*glm::pi<float> () * j * S) * sin( glm::pi<float> () * i * R );
        float y = sin( - glm::half_pi <float> ()  + glm::pi<float> () * i * R );
        float z = sin(2*glm::pi<float> ()* j * S) * sin( glm::pi<float> () * i * R );
		if (i==0)j= meridian;
		if(i==(paralel-1))j= meridian;
        // Push Back Vertex Data
        pos.push_back (glm::vec3 (x, y, z) * radius);
    }
}


}

std::string getexepath()
  {
      char buffer[MAX_PATH];
    GetModuleFileName( NULL, buffer, MAX_PATH );
    std::string::size_type pos = std::string( buffer ).find_last_of( "\\/" );
    return std::string( buffer ).substr( 0, pos);
  }
int  calculateviews(pcl::PolygonMesh mesh, bool seewindow,int width,int height, float radius_cam, float field_of_view, int meridian,int paralel,std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& views,std::vector<Eigen::Matrix4f>& poses )
{
	std::vector<glm::vec3>  campos;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr vertices_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::fromPCLPointCloud2 (mesh.cloud, *vertices_rgb);
	pcl::PointCloud<pcl::PointXYZ>::Ptr prueba (new pcl::PointCloud<pcl::PointXYZ>());
	Eigen::Vector4f min;
	Eigen::Vector4f max;
	// i need this because in pcl::MomentOfInertiaEstimation  the pcl::PointXYZRGB is not implemented
	for (int i =0;i<vertices_rgb->size();i++)
	{
		pcl::PointXYZ punto;
		punto.x = vertices_rgb->points[i].x;
		punto.y = vertices_rgb->points[i].y;
		punto.z = vertices_rgb->points[i].z;
		prueba->push_back(punto);
	}
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*prueba,centroid);
	 Eigen::Affine3f transform = Eigen::Affine3f::Identity();

	// Translate the centroid model to the origin.
	transform.translation() << -centroid[0], -centroid[1], -centroid[2];
	//save the translation, this will be used to restore the values
	pcl::transformPointCloud(*prueba,*prueba,transform);
	pcl::transformPointCloud(*vertices_rgb,*vertices_rgb,transform);
	pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud (prueba);
	feature_extractor.compute ();
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	feature_extractor.getAABB (min_point_AABB, max_point_AABB);

	float aux=0,aux2=0;
	//here the bounding box of the model is calculated
	float distancex = (max_point_AABB.x - min_point_AABB.x)/2;
	float distancey = (max_point_AABB.y - min_point_AABB.y)/2;
	float distancez = (max_point_AABB.z - min_point_AABB.z)/2;
	// calculate the point that is farther in the bounding box
	if (distancex>distancey) aux=distancex,aux2=distancey;
	else aux=distancey,aux2=distancex;
	if (aux<distancez)aux2= aux,aux = distancez;
	if (aux>distancez&&aux2<distancez) aux2=distancez;
	// we calculate the point that is more far from the center and scale to a unit sphere
	aux = std::sqrt((aux2*aux2)+(aux*aux));
	//apply the scale to fit the model in a unit sphere


	for (int i=0;i<vertices_rgb->size();i++)
	{
		vertices_rgb->points[i].x= vertices_rgb->points[i].x/aux;
		vertices_rgb->points[i].y= vertices_rgb->points[i].y/aux;
		vertices_rgb->points[i].z= vertices_rgb->points[i].z/aux;

	}

	// calculate the minimal distance of the cam_pos to enter all the object in the cam,
	//this value depends on the fov 0.5f is the half of the sphere 
	//                   1
	//               ______0.5
	//               \   |   /
	// minimal_radius \  |  /
	//                 \ | /
	//                  \|/   angle=fov/2
	//                 cam
	//                   
	float minimal_radius = 0.5f*std::cos((field_of_view/2)*M_PI/180 ) / std::sin((field_of_view/2)*M_PI/180);
	// add the sphere of the model
	minimal_radius+=1.0f;
	if (radius_cam < minimal_radius) radius_cam=minimal_radius;
		calculatecampoints(campos,minimal_radius,meridian,paralel);


	std::vector< glm::vec3 > vertices_dibujar_aux;
	std::vector< glm::vec3 > vertices_dibujar;
	std::vector< glm::vec3 > vertices_color;


	//introduce in a vector for opengl all the vertex of the mesh.
	for (int i =0;i<mesh.polygons.size();i++)
	{
		pcl::PointXYZRGB v1,v2,v3;
		v1= vertices_rgb->at(mesh.polygons[i].vertices[0]);
		v2= vertices_rgb->at(mesh.polygons[i].vertices[1]);
		v3= vertices_rgb->at(mesh.polygons[i].vertices[2]);

		vertices_dibujar.push_back(glm::vec3(v1.x,v1.y,v1.z));
		vertices_dibujar.push_back(glm::vec3(v2.x,v2.y,v2.z));
		vertices_dibujar.push_back(glm::vec3(v3.x,v3.y,v3.z));
		// to pass to float for opengl
		vertices_color.push_back(glm::vec3(v1.r/255.0f,v1.g/255.0f,v1.b/255.0f));
		vertices_color.push_back(glm::vec3(v2.r/255.0f,v2.g/255.0f,v2.b/255.0f));
		vertices_color.push_back(glm::vec3(v3.r/255.0f,v3.g/255.0f,v3.b/255.0f));


	}

	//// Initialise GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// to set if the window is visible or not.
	if(seewindow==false)glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	else glfwWindowHint(GLFW_VISIBLE, GL_TRUE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(width, height, "calculate views", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible.\n" );
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}




	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS); 

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);
	std::stringstream ss_cloud;

	 std::string direxe =getexepath();
	ss_cloud<<direxe<<"\\TransformVertexShader.vertexshader";
	std::stringstream ss_cloud2;
	ss_cloud2<<direxe<<"\\ColorFragmentShader.fragmentshader";
	// Create and compile our GLSL program from the shaders

	const std::string tmp = ss_cloud.str();
	const char* cstr = tmp.c_str();
	const std::string tmp2 = ss_cloud2.str();
	const char* cstr2 = tmp2.c_str();
	GLuint programID = LoadShaders(cstr, cstr2);

	// Get a handle for our "MVP" uniform


	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices_dibujar.size() * sizeof(glm::vec3), &vertices_dibujar[0], GL_STATIC_DRAW);


	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER,  vertices_color.size() * sizeof(glm::vec3), &vertices_color[0], GL_STATIC_DRAW);

	for (int a=0;a<campos.size();a++)
	{	


		pcl::PointCloud<pcl::PointXYZRGB> n_puntos_final;
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Use our shader
		glUseProgram(programID);

		GLuint MatrixID = glGetUniformLocation(programID, "MVP");
		// Use our shader
		float relation = 1.0*width / height;
		// the perspective is fixed, in the relation
		glm::mat4 Projection = glm::perspective(field_of_view, relation, 0.10f,radius_cam+ 1.0f);
		// Camera matrix // this is the pose to store(i think, test)
		glm::mat4 View    = glm::lookAt(
			campos[a], // 
			glm::vec3(0.0f,0.0f,0.0f), // and looks at the origin
			glm::vec3(0.0f,0.0f,1.0f)  
			);


		
		glm::mat4 Model = glm::scale(glm::mat4(1.0f),glm::vec3(1.0f, 1.0f, 1.0f));
		
		glm::mat4 Model_View= View * Model;
	    Eigen::Matrix4f pose_aux ;
		for(int i=0;i<4;i++)
		{
		for (int j=0;j<4;j++)
		{
		pose_aux(i,j)= static_cast<float>(Model_View[j][i]);
		
		}
		
		}

		pose_aux(0,3)= (static_cast<float>(Model_View[3][0])*aux);
		pose_aux(1,3)= static_cast<float>(Model_View[3][1])*aux;
		pose_aux(2,3)= static_cast<float>(Model_View[3][2])*aux;
		Eigen::Matrix4f Matrix_aux = transform.matrix();
		// add the translation of the centroid to the final mode
		// this is the pose of the view that is stored and output the program
		pose_aux =  pose_aux*Matrix_aux;
		poses.push_back(pose_aux);


		// Our ModelViewProjection : multiplication of our 3 matrices
		glm::mat4 MVP        = Projection * View * Model; // Remember, matrix multiplication is the other way around
		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);


		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // attribute. No part icular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
			);

		//// 2nd attribute buffer : colors
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
			);

		// Draw the triangles !
		glDrawArrays(GL_TRIANGLES, 0, vertices_dibujar.size() );

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		// grab depth buffer from OpenGL
		float depth_range[2];
		//GLfloat buffer[16];

		GLint view[4];
		glGetFloatv(GL_DEPTH_RANGE, depth_range);
		glGetIntegerv(GL_VIEWPORT, view);


		const float &dNear = depth_range[0];
		const float &dFar = depth_range[1];
		const float &fx = Projection[0][0];
		const float &fy = Projection[1][1];
		const float &cx = Projection[0][2];
		const float &cy = Projection[1][2];
		const float &z1 = Projection[2][2];
		const float &z2 = Projection[3][2];
		int width = view[2];
		int height = view[3];

		float zFar = z2 / (z1 + 1.0f);
		float zNear = z2 * zFar / (z2 - 2.0f * zFar);

		// grab depth buffer from OpenGL
		float *depthbuffer = (float*) malloc(sizeof(float) * width * height);
		glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depthbuffer);

		float* colorbuffer = new float[width*height*3];
		//float *colorbuffer = (float*) malloc(sizeof(float) * width * height);
		glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, colorbuffer);

		// parse depth values and create 3D points in world coordinates
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				int winY = j;
				winY = height - j - 1; // inverse y coordinates

				int idx = winY * width + i; // depthbuffer index
				const float &z_b = depthbuffer[idx];


				if (z_b > dNear && z_b < dFar) // depth range check
				{

				
        
					//float z = z_b;
					float z_n = 2.0 * z_b - 1.0; // normalized depth

					//// transform to camera coordinates
					float z = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
					float x = (2.0 * float(i) / width - 1.0f - cx) * z / fx;
					float y = (2.0 * float(winY) / height - 1.0f - cy) * z / fy;
					pcl::PointXYZRGB puntito;
					//transform to real world
					z=-z;
					y=y;
					x=x;

					puntito.x= (x*aux);
					puntito.y= y*aux;
					puntito.z= z*aux;
			
					puntito.r = colorbuffer[3*idx]*255;
					puntito.g = colorbuffer[(3*idx)+1]*255;
					puntito.b = colorbuffer[(3*idx)+2]*255;
					n_puntos_final.push_back(puntito);



				}
			}
		}


		pcl::PointCloud<pcl::PointXYZRGB>::Ptr final ( new pcl::PointCloud<pcl::PointXYZRGB>());
		final= n_puntos_final.makeShared();

		views.push_back(final);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	}

	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &colorbuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	return views.size();

}


void createreferencematrix (std::string path, Eigen::Matrix4f posmatrix)
{
	std::string delimiter = "  ";
	ofstream myfile (path);
	if (myfile.is_open())
	{
		for(int i =0;i<4;i++)
		{
			myfile<< posmatrix(i,0)<<"   "<<posmatrix(i,1)<<"   "<<posmatrix(i,2)<<"   "<<posmatrix(i,3)<<"\n";
		}	


		myfile.close();
	}
}


void
	parseCommandLine (int argc, char *argv[])
{
	//Show help
	if (pcl::console::find_switch (argc, argv, "-h"))
	{
		showHelp (argv[0]);
		exit (0);
	}

	//Model 
	std::vector<int> modelname;
	modelname = pcl::console::parse_file_extension_argument (argc, argv, ".ply");
	if (modelname.size () != 1)
	{
		std::cout << "Filenames missing.\n";
		showHelp (argv[0]);
		exit (-1);
	}

	model_filename_ = argv[modelname[0]];

	//General parameters
	pcl::console::parse_argument (argc, argv, "--width", width);
	pcl::console::parse_argument (argc, argv, "--height", height);
	pcl::console::parse_argument (argc, argv, "--radius", rad);
	pcl::console::parse_argument (argc, argv, "--fov",fov);
	pcl::console::parse_argument (argc, argv, "--visualizate", visualizate);
	pcl::console::parse_argument (argc, argv, "--meridian", merid);
	pcl::console::parse_argument (argc, argv, "--paralel", para);
	pcl::console::parse_argument (argc, argv, "--dir", dir);
}



int main (int argc, char *argv[])
{

	parseCommandLine (argc, argv);
	std::cout<<"directorio"<<dir<<std::endl;


	pcl::PolygonMesh mesh;
	std::vector<glm::vec3>  campos;
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>> nubes_vis;
	pcl::io::loadPolygonFilePLY(model_filename_,mesh);
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> views;
	std::vector<Eigen::Matrix4f> poses;


	if (calculateviews(mesh,false,width,height,rad,fov,merid,para,views,poses)<0)
	{

		std::cout<<"something was wrong, check the mesh or the parameters"<<std::endl;
		return -1;
	}
	//Visualize the results
	pcl::visualization::PCLVisualizer viewer ("See results");
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr allpoints (new pcl::PointCloud<pcl::PointXYZRGB>());

				
	for (size_t h = 0; h <views.size(); ++h)
	{


		std::stringstream ss_cloud;
		std::stringstream ss_cloud2;
		ss_cloud  <<dir<< h<<".pcd";
		ss_cloud2 <<dir<< "pose" << h<<".txt";
		pcl::io::savePCDFileBinary(ss_cloud.str(),*views[h]);
		createreferencematrix(ss_cloud2.str(),poses[h]);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr add (new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr in (new pcl::PointCloud<pcl::PointXYZRGB>());
		*in= *views[h];
		Eigen::Matrix4f tranform = poses[h].inverse();
		pcl::transformPointCloud(*in,*add,tranform);
		//viewer.addPointCloud(add,ss_cloud.str());

	}
	//while (!viewer.wasStopped ())
	//{
	//	viewer.spinOnce ();
	//}
	//viewer.close();	


	return 0;
}

