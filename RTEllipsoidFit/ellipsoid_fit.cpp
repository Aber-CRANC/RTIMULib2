
#include "ellipsoid_fit.h"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

int fit_ellipsoid(const char * filename_in, const char * filename_out)
{
	FILE * fp;
	double mag_x, mag_y, mag_z;
	long line_count = 0;
	long index = 0;

	MatrixXd mat_D;
	MatrixXd mat_DT;

	fp = fopen(filename_in, "r");

	if (fp == NULL)
	{
		printf("Error opening %s.\n", filename_in);
		exit(1);
	}

	//  check how many lines (data points) are in the file
	while(!feof(fp))
	{
		if(fgetc(fp) == '\n')
		{
			line_count++;
		}
	}

	mat_D = MatrixXd(line_count, 9);

	//  start reading the data points from the top of the file
	rewind(fp);

	while (fscanf(fp, "%lf %lf %lf\n", &mag_x, &mag_y, &mag_z) == 3) {
		mat_D(index, 0) = mag_x * mag_x;
		mat_D(index, 1) = mag_y * mag_y;
		mat_D(index, 2) = mag_z * mag_z;
		mat_D(index, 3) = 2 * mag_x * mag_y;
		mat_D(index, 4) = 2 * mag_x * mag_z;
		mat_D(index, 5) = 2 * mag_y * mag_z;
		mat_D(index, 6) = 2 * mag_x;
		mat_D(index, 7) = 2 * mag_y;
		mat_D(index, 8) = 2 * mag_z;

		index++;
	}

	fclose(fp);

	mat_DT = mat_D.transpose();

	MatrixXd mat_Ones = MatrixXd::Ones(index, 1);

	MatrixXd mat_Result =  (mat_DT * mat_D).inverse() * (mat_DT * mat_Ones);

	Matrix<double, 4, 4>  mat_A_4x4;

	mat_A_4x4(0, 0) = mat_Result(0, 0);
	mat_A_4x4(0, 1) = mat_Result(3, 0);
	mat_A_4x4(0, 2) = mat_Result(4, 0);
	mat_A_4x4(0, 3) = mat_Result(6, 0);

	mat_A_4x4(1, 0) = mat_Result(3, 0);
	mat_A_4x4(1, 1) = mat_Result(1, 0);
	mat_A_4x4(1, 2) = mat_Result(5, 0);
	mat_A_4x4(1, 3) = mat_Result(7, 0);

	mat_A_4x4(2, 0) = mat_Result(4, 0);
	mat_A_4x4(2, 1) = mat_Result(5, 0);
	mat_A_4x4(2, 2) = mat_Result(2, 0);
	mat_A_4x4(2, 3) = mat_Result(8, 0);

	mat_A_4x4(3, 0) = mat_Result(6, 0);
	mat_A_4x4(3, 1) = mat_Result(7, 0);
	mat_A_4x4(3, 2) = mat_Result(8, 0);
	mat_A_4x4(3, 3) = -1.0;

	MatrixXd mat_Center = -((mat_A_4x4.block(0, 0, 3, 3)).inverse() * mat_Result.block(6, 0, 3, 1));

	Matrix<double, 4, 4>  mat_T_4x4;
	mat_T_4x4.setIdentity();
	mat_T_4x4.block(3, 0, 1, 3) = mat_Center.transpose();

	MatrixXd mat_R = mat_T_4x4 * mat_A_4x4 * mat_T_4x4.transpose();

	EigenSolver<MatrixXd> eig(mat_R.block(0, 0, 3, 3) / -mat_R(3, 3));
	//mat_T_4x4(3,0)=mat_Center()
	MatrixXd mat_Eigval(3, 1) ;
	MatrixXd mat_Evecs(3, 3) ;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mat_Evecs(i, j) = (eig.eigenvectors())(i, j).real();
		}
	}

	mat_Eigval(0, 0) = (eig.eigenvalues())(0, 0).real();
	mat_Eigval(1, 0) = (eig.eigenvalues())(1, 0).real();
	mat_Eigval(2, 0) = (eig.eigenvalues())(2, 0).real();
	MatrixXd mat_Radii = (1.0 / mat_Eigval.array()).cwiseSqrt();
	MatrixXd mat_Scale = MatrixXd::Identity(3, 3) ;
	mat_Scale(0, 0) = mat_Radii(0, 0);
	mat_Scale(1, 1) = mat_Radii(1, 0);
	mat_Scale(2, 2) = mat_Radii(2, 0);
	double min_Radii = mat_Radii.minCoeff();

	mat_Scale = mat_Scale.inverse().array() * min_Radii;
	MatrixXd mat_Correct = mat_Evecs * mat_Scale * mat_Evecs.transpose();

	cout << "Ellipsoid fit done using " << index << " points out " << line_count << " data points (lines) in the file" << endl;
	cout << "The Ellipsoid center is:" << '\n' << mat_Center << endl;
	cout << "The Ellipsoid radii is:" << '\n' << mat_Radii << endl;
	cout << "The scale matrix  is:" << '\n' << mat_Scale << endl;
	cout << "The correct matrix  is:" << '\n' << mat_Correct << endl;

	FILE *correction_file = fopen(filename_out, "w");
	if (correction_file == NULL)
	{
		printf("Error opening %s\n", filename_out);
		exit(1);
	}

	fprintf(correction_file, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
	mat_Center(0), mat_Center(1), mat_Center(2),
	mat_Correct(0,0), mat_Correct(0,1), mat_Correct(0,2),
	mat_Correct(1,0), mat_Correct(1,1), mat_Correct(1,2),
	mat_Correct(2,0), mat_Correct(2,1), mat_Correct(2,2));

	fclose(correction_file);

	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Too few arguments. Usage:\n");
		printf("ellipsoid_fit [input_file] [output_file]\n");
		return 1;
	}

	if (!fit_ellipsoid(argv[1], argv[2])) {
		return 1;
	}

	return 0;
}
