#pragma once

#ifdef _WIN32
  #ifdef CUDALIB_EXPORTS
    #define CUDA_API __declspec(dllexport)  // Экспорт для DLL
  #else
    #define CUDA_API __declspec(dllimport)  // Импорт для DLL
  #endif
#else
  #define CUDA_API  // Для других систем
#endif
// -----------------------
// --- ���������� CUDA ---
// -----------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// -----------------------

// --------------------------------------------
// --- KiShiVi ���������� ��� ������ � CUDA ---
// --------------------------------------------

#include "cudaMacros.cuh"
#include "cudaLibrary.cuh"

// --------------------------------------------

// -----------------------------
// --- ���������� ���������� ---
// -----------------------------

#include <iomanip>
#include <string>

// -----------------------------



/**
 * �������, ��� ������� ���������� �������������� ���������.
 */
CUDA_API __host__ void distributedSystemSimulation(
	const double	tMax,							// ����� ������������� �������
	const double	h,								// ��� ��������������
	const double	hSpecial,						// ��� �������� ����� ��������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues);				// ���������� ����������				



/**
 * �������, ��� ������� ���������� �������������� ���������.
 */
CUDA_API __host__ void bifurcation1D(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const double	h,								// ��� ��������������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const double*	ranges,							// �������� ��������� ����������
	const int*		indicesOfMutVars,				// ������ ���������� ���������� � ������� values
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller);					// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)



/**
 * �������, ��� ������� ���������� �������������� ��������� �� ����.
 */
CUDA_API __host__ void bifurcation1DForH(
	const double	tMax,							// ����� ������������� �������
	const int		nPts,							// ���������� ���������
	const int		amountOfInitialConditions,		// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				// ������ � ���������� ���������
	const double*	ranges,							// �������� ��������� ����
	const int		writableVar,					// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							// ���������
	const int		amountOfValues,					// ���������� ����������
	const int		preScaller);					// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)



/**
 * �������, ��� ������� ���������� �������������� ���������. (�� ��������� ��������)
 */
CUDA_API __host__ void bifurcation1DIC(
	const double	tMax,							  // ����� ������������� �������
	const int		nPts,							  // ���������� ���������
	const double	h,								  // ��� ��������������
	const int		amountOfInitialConditions,		  // ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,				  // ������ � ���������� ���������
	const double*	ranges,							  // �������� ��������� ���������� �������
	const int*		indicesOfMutVars,				  // ������ ����������� ���������� �������
	const int		writableVar,					  // ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,						  // ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,					  // �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,							  // ���������
	const int		amountOfValues,					  // ���������� ����������
	const int		preScaller);					  // ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)



/**
 * �������, ��� ������� ��������� �������������� ��������� (DBSCAN)
 */
CUDA_API __host__ void bifurcation2D(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,					// ������ � ���������� ���������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps);								// ������� ��� ��������� DBSCAN 



/**
 * �������, ��� ������� ��������� �������������� ��������� (DBSCAN) (for initial conditions)
 */
CUDA_API __host__ void bifurcation2DIC(
	const double	tMax,								// ����� ������������� �������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	initialConditions,					// ������ � ���������� ���������
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues,						// ���������� ����������
	const int		preScaller,							// ���������, ������� ��������� ����� � ����� �������� (����� �������������� ������ ������ 'preScaller' �����)
	const double	eps);								// ������� ��� ��������� DBSCAN 



/**
 * ���������� 1D LLE ���������
 */
CUDA_API __host__ void LLE1D(
	const double	tMax,								// ����� ������������� �������
	const double	NT,									// ����� ������������
	const int		nPts,								// ���������� ���������
	const double	h,									// ��� ��������������
	const double	eps,								// ������� ��� LLE
	const double*	initialConditions,					// ������ � ���������� ���������
	const int		amountOfInitialConditions,			// ���������� ��������� ������� ( ��������� � ������� )
	const double*	ranges,								// ��������� ��������� ����������
	const int*		indicesOfMutVars,					// ������� ���������� ����������
	const int		writableVar,						// ������ ���������, �� �������� ����� ������� ���������
	const double	maxValue,							// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double	transientTime,						// �����, ������� ����� ��������������� ����� �������� ���������
	const double*	values,								// ���������
	const int		amountOfValues);					// ���������� ����������



/**
 * ���������� 1D LLE ��������� (IC)
 */
CUDA_API __host__ void LLE1DIC(			
	const double tMax,									// ����� ������������� �������
	const double NT,									// ����� ������������
	const int nPts,										// ���������� ���������
	const double h,										// ��� ��������������
	const double eps,									// ������� ��� LLE
	const double* initialConditions,					// ������ � ���������� ���������
	const int amountOfInitialConditions,				// ���������� ��������� ������� ( ��������� � ������� )
	const double* ranges,								// ��������� ��������� ����������
	const int* indicesOfMutVars,						// ������� ���������� ����������
	const int writableVar,								// ������ ���������, �� �������� ����� ������� ���������
	const double maxValue,								// ������������ �������� (�� ������), ���� �������� ������� ��������� "�����������"
	const double transientTime,							// �����, ������� ����� ��������������� ����� �������� ���������
	const double* values,								// ���������
	const int amountOfValues);							// ���������� ����������



/**
 * Construction of a 2D LLE diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
CUDA_API __host__ void LLE2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);



/**
 * Construction of a 2D LLE diagram (for initial conditions)
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
CUDA_API __host__ void LLE2DIC(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);



/**
 * Construction of a 1D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
CUDA_API __host__ void LS1D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);




/**
 * Construction of a 2D LS diagram
 *
 * \param tMax - Simulation time
 * \param NT - Normalization time
 * \param nPts - Resolution
 * \param h - Integration step
 * \param eps - Eps
 * \param initialConditions - Array of initial conditions
 * \param amountOfInitialConditions - Amount of initial conditions
 * \param ranges - Array with variable parameter ranges
 * \param indicesOfMutVars - Index of unknown variable
 * \param writableVar - Evaluation axis (X - 0, Y - 1, Z - 2)
 * \param maxValue - Threshold signal level
 * \param transientTime - Transient time
 * \param values - Array of parameters
 * \param amountOfValues - Amount of Parameters
 * \return -
 */
CUDA_API __host__ void LS2D(
	const double tMax,
	const double NT,
	const int nPts,
	const double h,
	const double eps,
	const double* initialConditions,
	const int amountOfInitialConditions,
	const double* ranges,
	const int* indicesOfMutVars,
	const int writableVar,
	const double maxValue,
	const double transientTime,
	const double* values,
	const int amountOfValues);
