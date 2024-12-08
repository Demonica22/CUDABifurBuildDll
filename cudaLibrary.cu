#include "cudaLibrary.cuh"

// ---------------------------------------------------------------------------------
// --- Âû÷èñëÿåò ñëåäóþùåå çíà÷åíèå äèñêðåòíîé ìîäåëè è çàïèñûâàåò ðåçóëüòàò â x ---
// ---------------------------------------------------------------------------------

__device__ __host__ void calculateDiscreteModel(double* x, const double* a, const double h)
{
	/**
	 * here we abstract from the concept of parameter names.
	 * ALL parameters are numbered with indices.
	 * In the current example, the parameters go like this:
	 * 
	 * values[0] - sym
	 * values[1] - A
	 * values[2] - B
	 * values[3] - C
	 */

	double h1 = h * a[0];
	double h2 = h * (1 - a[0]);
	x[0] = x[0] + h1 * (-x[1] - x[2]);
	x[1] = (x[1] + h1 * (x[0])) / (1 - a[1] * h1);
	x[2] = (x[2] + h1 * a[2]) / (1 - h1 * (x[0] - a[3]));
	x[2] = x[2] + h2 * (a[2] + x[2] * (x[0] - a[3]));
	x[1] = x[1] + h2 * (x[0] + a[1] * x[1]);
	x[0] = x[0] + h2 * (-x[1] - x[2]);

}


// -----------------------------------------------------------------------------------------------------
// --- Âû÷èñëÿåò òðàåêòîðèþ äëÿ îäíîé ñèñòåìû è çàïèñûâàåò ðåçóëüòàò â "data" (åñëè data != nullptr) ---
// -----------------------------------------------------------------------------------------------------

__device__ __host__ bool loopCalculateDiscreteModel(double* x, const double* values, 
	const double h, const int amountOfIterations, const int amountOfX, const int preScaller,
	int writableVar, const double maxValue, double* data, 
	const int startDataIndex, const int writeStep)
{
	double* xPrev = new double[amountOfX];
	// --- Ãëîáàëüíûé öèêë, êîòîðûé ïðîèçâîäèò âû÷èñëåíèÿ çàäàííûå amountOfIterations ðàç ---
	for ( int i = 0; i < amountOfIterations; ++i )
	{
		for (int j = 0; j < amountOfX; ++j)
		{
			xPrev[j] = x[j];
		}
		// --- Åñëè âñå-òàêè ïåðåäàëè ìàññèâ äëÿ çàïèñè - çàïèñûâàåì çíà÷åíèå ïåðåìåííîé ---
		if ( data != nullptr )
			data[startDataIndex + i * writeStep] = x[writableVar];

		// --- Ìîäåëèðóåì ñèñòåìó preScaller ðàç ( òî åñòü åñëè preScaller > 1, òî ìû ïðîïóñòèì ( preScaller - 1 ) â ñìîäåëèðîâàííîé òðàåêòîðèè ) ---
		for ( int j = 0; j < preScaller; ++j )
			calculateDiscreteModel(x, values, h);

		// --- Åñëè isnan èëè isinf - âîçâðàùàåì false, èáî ýòî íåæåëàòåëüíîå ïîâåäåíèå ñèñòåìû ---
		if ( isnan( x[writableVar] ) || isinf( x[writableVar] ) )
		{
			delete[] xPrev;
			return false;
		}

		// --- Åñëè maxValue == 0, ýòî çíà÷èò ïîëüçîâàòåëü íå âûñòàâèë îãðàíè÷åíèå, èíà÷å òðåáóåòñÿ åãî ïðîâåðèòü ---
		if ( maxValue != 0 )
			if ( fabsf( x[writableVar] ) > maxValue )
			{
				delete[] xPrev;
				return false;
			}
	}

	// --- Ïðîâåðêà íà ñâàëèâàíèå â òî÷êó ---
	double tempResult = 0;
	for (int j = 0; j < amountOfX; ++j)
	{
		tempResult += ((x[j] - xPrev[j]) * (x[j] - xPrev[j]));
	}

	if (tempResult == 0)
	{
		delete[] xPrev;
		return false;
	}

	if (sqrt(tempResult) < 1e-12)
	{
		delete[] xPrev;
		return false;
	}

	delete[] xPrev;
	return true;
}



__global__ void distributedCalculateDiscreteModelCUDA(
	const int		amountOfPointsForSkip,
	const int		amountOfThreads,
	const double	h,
	const double	hSpecial,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		writableVar,
	double*			data)
{
	extern __shared__ double s[];
	double* localX = s + (threadIdx.x * amountOfInitialConditions);
	double* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfThreads)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Ïðîãîíÿåì ñèñòåìó amountOfPointsForSkip ðàç ( äëÿ îòðàáîòêè transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0);

	loopCalculateDiscreteModel(localX, localValues, h, idx,
		amountOfInitialConditions, 1, 0, 0, nullptr, 0, 0);

	loopCalculateDiscreteModel(localX, localValues, hSpecial, amountOfIterations,
		amountOfInitialConditions, 1, writableVar, 0, data, idx, amountOfThreads);

	return;
}



// --------------------------------------------------------------------------
// --- Ãëîáàëüíàÿ ôóíêöèÿ, êîòîðàÿ âû÷èñëÿåò òðàåêòîðèþ íåñêîëüêèõ ñèñòåì ---
// --------------------------------------------------------------------------

__global__ void calculateDiscreteModelCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const int		sizeOfBlock, 
	const int		amountOfCalculatedPoints, 
	const int		amountOfPointsForSkip,
	const int		dimension, 
	double*			ranges, 
	const double	h,
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const int		amountOfInitialConditions, 
	const double*	values, 
	const int		amountOfValues,
	const int		amountOfIterations, 
	const int		preScaller,
	const int		writableVar, 
	const double	maxValue, 
	double*			data, 
	int*			maxValueCheckerArray)
{
	// --- Îáùàÿ ïàìÿòü â ðàìêàõ îäíîãî áëîêà ---
	// --- Ñòðîåíèå ïàìÿòè: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ñëåäóþøèé ïîòîê...} ---
	extern __shared__ double s[];

	// --- Â êàæäîì ïîòîêå ñîçäàåì óêàçàòåëü íà ïàðàìåòðû è ïåðåìåííûå, ÷òîáû ðàáîòàòü ñ íèìè êàê ñ ìàññèâàìè ---
	double* localX = s + ( threadIdx.x * amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	// --- Âû÷èñëÿåì èíäåêñ ïîòîêà, â êîòîðîì íàõîäèìñÿ â äàíûé ìîìåíò ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Åñëè ñóùåñòâóåò ïîòîê ñ áîëüøèì èíäåêñîì, ÷åì òðåáóåòñÿ - ñðàçó çàâåðøàåì åãî
		return;

	// --- Îïðåäåëÿåì localX[] íà÷àëüíûìè óñëîâèÿìè ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- Îïðåäåëÿåì localValues[] íà÷àëüíûìè ïàðàìåòðàìè ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Ìåíÿåì çíà÷åíèå èçìåíÿåìûõ ïàðàìåòðîâ íà ðåçóëüòàò ôóíêöèè getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	// --- Ïðîãîíÿåì ñèñòåìó amountOfPointsForSkip ðàç ( äëÿ îòðàáîòêè transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		1, amountOfInitialConditions, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Òåïåðü óæå ïî-âçðîñëîìó ìîäåëèðóåì ñèñòåìó --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Åñëè ôóíêöèÿ ìîäåëèðîâàíèÿ âûäàëà false - çíà÷èò ìû äàæå íå áóäåì ñìîòðåòü íà ýòó ñèñòåìó â äàëüíåéøåì àíàëèçå ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;	

	return;
}



__global__ void calculateDiscreteModelCUDA_H(
	const int		nPts,
	const int		nPtsLimiter,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const double	transientTime,
	const int		dimension,
	double*			ranges,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const double	tMax,
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			data,
	int*			maxValueCheckerArray)
{
	// --- Îáùàÿ ïàìÿòü â ðàìêàõ îäíîãî áëîêà ---
	// --- Ñòðîåíèå ïàìÿòè: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ñëåäóþøèé ïîòîê...} ---
	extern __shared__ double s[];

	// --- Â êàæäîì ïîòîêå ñîçäàåì óêàçàòåëü íà ïàðàìåòðû è ïåðåìåííûå, ÷òîáû ðàáîòàòü ñ íèìè êàê ñ ìàññèâàìè ---
	double* localX = s + (threadIdx.x * amountOfInitialConditions);
	double* localValues = s + (blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	// --- Âû÷èñëÿåì èíäåêñ ïîòîêà, â êîòîðîì íàõîäèìñÿ â äàíûé ìîìåíò ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Åñëè ñóùåñòâóåò ïîòîê ñ áîëüøèì èíäåêñîì, ÷åì òðåáóåòñÿ - ñðàçó çàâåðøàåì åãî
		return;

	// --- Îïðåäåëÿåì localX[] íà÷àëüíûìè óñëîâèÿìè ---
	for (int i = 0; i < amountOfInitialConditions; ++i)
		localX[i] = initialConditions[i];

	// --- Îïðåäåëÿåì localValues[] íà÷àëüíûìè ïàðàìåòðàìè ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	//// --- Ìåíÿåì çíà÷åíèå èçìåíÿåìûõ ïàðàìåòðîâ íà ðåçóëüòàò ôóíêöèè getValueByIdx ---
	//for (int i = 0; i < dimension; ++i)
	//	localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
	//		nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	double h = pow(10, getValueByIdxLog(amountOfCalculatedPoints + idx, nPts, ranges[0], ranges[1], 0));

	// --- Ïðîãîíÿåì ñèñòåìó amountOfPointsForSkip ðàç ( äëÿ îòðàáîòêè transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, transientTime / h,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Òåïåðü óæå ïî-âçðîñëîìó ìîäåëèðóåì ñèñòåìó --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, tMax / h / preScaller,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Åñëè ôóíêöèÿ ìîäåëèðîâàíèÿ âûäàëà false - çíà÷èò ìû äàæå íå áóäåì ñìîòðåòü íà ýòó ñèñòåìó â äàëüíåéøåì àíàëèçå ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;
	else
		maxValueCheckerArray[idx] = tMax / h / preScaller;

	return;
}



__global__ void calculateDiscreteModelICCUDA(
	const int		nPts, 
	const int		nPtsLimiter, 
	const int		sizeOfBlock, 
	const int		amountOfCalculatedPoints, 
	const int		amountOfPointsForSkip,
	const int		dimension, 
	double*			ranges, 
	const double	h,
	int*			indicesOfMutVars, 
	double*			initialConditions,
	const int		amountOfInitialConditions, 
	const double*	values, 
	const int		amountOfValues,
	const int		amountOfIterations, 
	const int		preScaller,
	const int		writableVar, 
	const double	maxValue, 
	double*			data, 
	int*			maxValueCheckerArray)
{
	// --- Îáùàÿ ïàìÿòü â ðàìêàõ îäíîãî áëîêà ---
	// --- Ñòðîåíèå ïàìÿòè: ---
	// --- {localX_0, localX_1, localX_2, ..., localValues_0, localValues_1, ..., ñëåäóþøèé ïîòîê...} ---
	extern __shared__ double s[];

	// --- Â êàæäîì ïîòîêå ñîçäàåì óêàçàòåëü íà ïàðàìåòðû è ïåðåìåííûå, ÷òîáû ðàáîòàòü ñ íèìè êàê ñ ìàññèâàìè ---
	double* localX = s + ( threadIdx.x * amountOfInitialConditions );
	double* localValues = s + ( blockDim.x * amountOfInitialConditions ) + ( threadIdx.x * amountOfValues );

	// --- Âû÷èñëÿåì èíäåêñ ïîòîêà, â êîòîðîì íàõîäèìñÿ â äàíûé ìîìåíò ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPtsLimiter)		// Åñëè ñóùåñòâóåò ïîòîê ñ áîëüøèì èíäåêñîì, ÷åì òðåáóåòñÿ - ñðàçó çàâåðøàåì åãî
		return;

	// --- Îïðåäåëÿåì localX[] íà÷àëüíûìè óñëîâèÿìè ---
	for ( int i = 0; i < amountOfInitialConditions; ++i )
		localX[i] = initialConditions[i];

	// --- Îïðåäåëÿåì localValues[] íà÷àëüíûìè ïàðàìåòðàìè ---
	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	// --- Ìåíÿåì çíà÷åíèå èçìåíÿåìûõ ïàðàìåòðîâ íà ðåçóëüòàò ôóíêöèè getValueByIdx ---
	for (int i = 0; i < dimension; ++i)
		localX[indicesOfMutVars[i]] = getValueByIdx( amountOfCalculatedPoints + idx, 
			nPts, ranges[i * 2], ranges[i * 2 + 1], i );

	// --- Ïðîãîíÿåì ñèñòåìó amountOfPointsForSkip ðàç ( äëÿ îòðàáîòêè transientTime ) --- 
	loopCalculateDiscreteModel(localX, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, 0, nullptr, idx * sizeOfBlock);

	// --- Òåïåðü óæå ïî-âçðîñëîìó ìîäåëèðóåì ñèñòåìó --- 
	bool flag = loopCalculateDiscreteModel(localX, localValues, h, amountOfIterations,
		amountOfInitialConditions, preScaller, writableVar, maxValue, data, idx * sizeOfBlock);

	// --- Åñëè ôóíêöèÿ ìîäåëèðîâàíèÿ âûäàëà false - çíà÷èò ìû äàæå íå áóäåì ñìîòðåòü íà ýòó ñèñòåìó â äàëüíåéøåì àíàëèçå ---
	if (!flag && maxValueCheckerArray != nullptr)
		maxValueCheckerArray[idx] = -1;	

	return;
}


// --- Ôóíêöèÿ, êîòîðàÿ íàõîäèò èíäåêñ â ïîñëåäîâàòåëüíîñòè çíà÷åíèé ---
__device__ __host__ double getValueByIdx(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return startRange + ( ( ( int )( ( int )idx / powf( ( double )nPts, ( double )valueNumber) ) % nPts )
		* ( ( double )( finishRange - startRange ) / ( double )( nPts - 1 ) ) );
}



// --- Ôóíêöèÿ, êîòîðàÿ íàõîäèò èíäåêñ â ïîñëåäîâàòåëüíîñòè çíà÷åíèé ---
__device__ __host__ double getValueByIdxLog(const int idx, const int nPts,
	const double startRange, const double finishRange, const int valueNumber)
{
	return log10(startRange) + (((int)((int)idx / powf((double)nPts, (double)valueNumber)) % nPts)
		* ((double)(log10(finishRange) - log10(startRange)) / (double)(nPts - 1)));
}



// ---------------------------------------------------------------------------------------------------
// --- Íàõîäèò ïèêè â èíòåðâàëå [startDataIndex; startDataIndex + amountOfPoints] â "data" ìàññèâå ---
// ---------------------------------------------------------------------------------------------------

__device__ __host__ int peakFinder(double* data, const int startDataIndex, 
	const int amountOfPoints, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- Ïåðåìåííàÿ äëÿ õðàíåíèÿ íàéäåííûõ ïèêîâ ---
	int amountOfPeaks = 0;

	// --- Íà÷èíàåì ïðîñìàòðèâàòü çàäàííûõ èíòåðâàë íà íàëè÷èå ïèêîâ ---
	for ( int i = startDataIndex + 1; i < startDataIndex + amountOfPoints - 1; ++i )
	{
		// --- Åñëè òåêóùàÿ òî÷êà áîëüøå ïðåäûäóùåé è áîëüøå ÈËÈ ÐÀÂÍÀ ñëåäóþùåé, òî... ( íå ôàêò, ÷òî ýòî ïèê ( íàïðèìåð: 2 3 3 4 ) ) ---
		if ( data[i] > data[i - 1] && data[i] >= data[i + 1] )
		{
			// --- Îò íàéäåííîé òî÷êè íà÷èíàåì èäòè âïåðåä, ïîêà íå íàòêíåìñÿ íà òî÷êó ñòðîãî áîëüøå èëè ìåíüøå ---
			for ( int j = i; j < startDataIndex + amountOfPoints - 1; ++j )
			{
				// --- Åñëè íàòêíóëèñü íà òî÷êó ñòðîãî áîëüøå, çíà÷èò ýòî áûë íå ïèê ---
				if ( data[j] < data[j + 1] )
				{
					i = j + 1;	// --- Îáíîâëÿåì âíåøíèé ñ÷åò÷èê, ÷òîáû äâàæäû íå ïðîõîäèòü îäèí è òîò æå èíòåðâàë
					break;		// --- Âîçâðàùàåìñÿ ê âíåøíåìó öèêëó
				}
				// --- Åñëè î ÷óäî, ìû íàøëè òî÷êó ìåíüøå, ÷åì òåêóùàÿ, çíà÷èò ìû íàøëè ïèê ---
				if ( data[j] > data[j + 1] )
				{
					// --- Åñëè ìàññèê outPeaks íå ïóñò, òî äåëàåì çàïèñü ---
					if ( outPeaks != nullptr )
						outPeaks[startDataIndex + amountOfPeaks] = data[j];
					// --- Åñëè ìàññèê timeOfPeaks íå ïóñò, òî äåëàåì çàïèñü ---
					if ( timeOfPeaks != nullptr )
						timeOfPeaks[startDataIndex + amountOfPeaks] = trunc( ( (double)j + (double)i ) / (double)2 );	// Âûáèðàåì èíäåêñ ïîñåðåäèíå ìåæäó j è i
					++amountOfPeaks;
					i = j + 1; // Ïîòîìó ÷òî ñëåäóþùàÿ òî÷êà òî÷íî íå ìîæåò áûòü ïèêîì ( äâà ïèêà íå ìîãóò èäòè ïîäðÿä )
					break;
				}
			}
		}
	}
	// --- Âû÷èñëÿåì ìåæïèêîâûå èíòåðâàëû ---
	if ( amountOfPeaks > 1 ) {
		// --- Ïðîáåãàåìñÿ ïî âñåì íàéäåííûì ïèêàì è èõ èíäåêñàì ---
		for ( size_t i = 0; i < amountOfPeaks - 1; i++ )
		{
			// --- Ñìåùàåì âñå ïèêè íà îäèí èíäåêñ âëåâî, à ïåðâûé ïèê óäàëÿåì ---
			if ( outPeaks != nullptr )
				outPeaks[startDataIndex + i] = outPeaks[startDataIndex + i + 1];
			// --- Âû÷èñëÿåì ìåæïèêîâûé èíòåðâàë. Ýòî ðàçíèöà èíäåêñà ñëåäóþùåãî ïðèêà è ïðåäûäóùåãî, óìíîæåííàÿ íà øàã ---
			if ( timeOfPeaks != nullptr )
				timeOfPeaks[startDataIndex + i] = ( double )( ( timeOfPeaks[startDataIndex + i + 1] - timeOfPeaks[startDataIndex + i] ) * h );
		}
		// --- Òàê êàê îäèí ïèê óäàëèëè - âû÷èòàåì åäèíèöó èç ðåçóëüòàòà ---
		amountOfPeaks = amountOfPeaks - 1;
	}
	else {
		amountOfPeaks = 0;
	}


	return amountOfPeaks;
}



// ----------------------------------------------------------------
// --- Íàõîæäåíèå ïèêîâ â "data" ìàññèâå â ìíîãîïîòî÷íîì ðåæèìå ---
// ----------------------------------------------------------------

__global__ void peakFinderCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks, 
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- Âû÷èñëÿåì èíäåêñ ïîòîêà, â êîòîðîì íàõîäèìñÿ â äàíûé ìîìåíò ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( idx >= amountOfBlocks )		// Åñëè ñóùåñòâóåò ïîòîê ñ áîëüøèì èíäåêñîì, ÷åì òðåáóåòñÿ - ñðàçó çàâåðøàåì åãî
		return;

	// --- Åñëè íà ïðåäûäóùèõ ýòàïàõ ñèñòåìó óæå îòìåòèëè êàê "íåïðèãîäíóþ", òî ïðîïóñêàåì åå ---
	if ( amountOfPeaks[idx] == -1 )
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	
	amountOfPeaks[idx] = peakFinder( data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks, h );
	return;
}



__global__ void peakFinderCUDA_H(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, double h)
{
	// --- Âû÷èñëÿåì èíäåêñ ïîòîêà, â êîòîðîì íàõîäèìñÿ â äàíûé ìîìåíò ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Åñëè ñóùåñòâóåò ïîòîê ñ áîëüøèì èíäåêñîì, ÷åì òðåáóåòñÿ - ñðàçó çàâåðøàåì åãî
		return;

	// --- Åñëè íà ïðåäûäóùèõ ýòàïàõ ñèñòåìó óæå îòìåòèëè êàê "íåïðèãîäíóþ", òî ïðîïóñêàåì åå ---
	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		return;
	}

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, amountOfPeaks[idx], outPeaks, timeOfPeaks, h);
	return;
}



__global__ void peakFinderCUDAForCalculationOfPeriodicityByOstrovsky(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, double* outPeaks, double* timeOfPeaks, bool* flags, double ostrovskyThreshold)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		amountOfPeaks[idx] = 0;
		flags[idx * 5 + 3] = true;
		return;
	}

	double lastPoint = data[idx * sizeOfBlock + sizeOfBlock - 1];

	amountOfPeaks[idx] = peakFinder(data, idx * sizeOfBlock, sizeOfBlock, outPeaks, timeOfPeaks);

	//FIRST CONDITION
	flags[idx * 5 + 0] = true;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] - outPeaks[i - 1] > 0)
		{
			flags[idx * 5 + 0] = false;
			break;
		}
	}

	//SECOND & THIRD CONDITION
	bool flagOne = false;
	bool flagZero = false;
	for (int i = idx * sizeOfBlock + 1; i < idx * sizeOfBlock + amountOfPeaks[idx]; ++i)
	{
		if (outPeaks[i] > ostrovskyThreshold)
			flagOne = true;
		else
			flagZero = true;
		if (flagOne && flagZero)
			break;
	}

	if (flagOne && flagZero)
		flags[idx * 5 + 1] = true;
	else
		flags[idx * 5 + 1] = false;

	if (flagOne && !flagZero)
		flags[idx * 5 + 2] = false;
	else
		flags[idx * 5 + 2] = true;

	//FOUR CONDITION
	if (amountOfPeaks[idx] == 0 || amountOfPeaks[idx] == 1)
		flags[idx * 5 + 3] = true;
	else
		flags[idx * 5 + 3] = false;

	//FIVE CONDITION
	if (lastPoint > ostrovskyThreshold)
		flags[idx * 5 + 4] = true;
	else
		flags[idx * 5 + 4] = false;
	return;
}



__device__ __host__ int kde(double* data, const int startDataIndex, const int amountOfPoints,
	int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH)
{
	if (amountOfPoints == 0)
		return 0;
	if (amountOfPoints == 1 || amountOfPoints == 2)
		return 1;
	if (amountOfPoints > maxAmountOfPeaks)
		return maxAmountOfPeaks;

	double k1 = kdeSampling * amountOfPoints;
	double k2 = (kdeSamplesInterval2 - kdeSamplesInterval1) / (k1 - 1);
	double delt = 0;
	double prevPrevData2 = 0;
	double prevData2 = 0;
	double data2 = 0;
	bool strangePeak = false;
	int resultKde = 0;

	for (int w = 0; w < k1 - 1; ++w)
	{
		delt = w * k2 + kdeSamplesInterval1;
		prevPrevData2 = prevData2;
		prevData2 = data2;
		data2 = 0;
		for (int m = 0; m < amountOfPoints; ++m)
		{
			double tempData = (data[startDataIndex + m] - delt) / kdeSmoothH;
			data2 += expf(-((tempData * tempData) / 2));
		}

		if (w < 2)
			continue;
		if (strangePeak)
		{
			if (prevData2 == data2)
				continue;
			else if (prevData2 < data2)
			{
				strangePeak = false;
				continue;
			}
			else if (prevData2 > data2)
			{
				strangePeak = false;
				++resultKde;
				continue;
			}
		}
		else if (prevData2 > prevPrevData2 && prevData2 > data2)
		{
			++resultKde;
			continue;
		}
		else if (prevData2 > prevPrevData2 && prevData2 == data2)
		{
			strangePeak = true;
			continue;
		}
	}
	if (prevData2 < data2)
	{
		++resultKde;
	}
	return resultKde;
}



__global__ void kdeCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	int* amountOfPeaks, int* kdeResult, int maxAmountOfPeaks, int kdeSampling, double kdeSamplesInterval1,
	double kdeSamplesInterval2, double kdeSmoothH)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)
		return;

	if (amountOfPeaks[idx] == -1)
	{
		kdeResult[idx] = 0;
		return;
	}
	kdeResult[idx] = kde(data, idx * sizeOfBlock, amountOfPeaks[idx], maxAmountOfPeaks,
		kdeSampling, kdeSamplesInterval1, kdeSamplesInterval2, kdeSmoothH);
}


// ------------------------------------------------
// --- Âû÷èñëÿåò ðàññòîÿíèå ìåæäó äâóìÿ òî÷êàìè ---
// ------------------------------------------------

__device__ __host__ double distance(double x1, double y1, double x2, double y2)
{
	if (x1 == x2 && y1 == y2)
		return 0;
	double dx = x2 - x1;
	double dy = y2 - y1;

	return hypotf(dx, dy);
}



// ----------------------
// --- Ôóíêöèÿ DBSCAN ---
// ----------------------

__device__ __host__ int dbscan(double* data, double* intervals, double* helpfulArray, 
	const int startDataIndex, const int amountOfPeaks, const int sizeOfHelpfulArray,
	const int idx, const double eps, int* outData)
{
	// ------------------------------------------------------------
	// --- Åñëè ïèêîâ 0 èëè 1 - äàæå íå îáðàáàòûâàåì ýòè ñëó÷àè ---
	// ------------------------------------------------------------

	if (amountOfPeaks <= 0)
		return 0;

	if (amountOfPeaks == 1)
		return 1;

	// ------------------------------------------------------------

	int cluster = 0;
	int NumNeibor = 0;

	for (int i = startDataIndex; i < startDataIndex + sizeOfHelpfulArray; ++i) {
		helpfulArray[i] = 0;
	}

	for (int i = 0; i < amountOfPeaks; i++)
		if (NumNeibor >= 1)
		{
			i = helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1];
			helpfulArray[startDataIndex + amountOfPeaks + NumNeibor - 1] = 0;
			NumNeibor = NumNeibor - 1;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
			}
		}
		else if (helpfulArray[startDataIndex + i] == 0) {
			NumNeibor = 0;
			++cluster;
			helpfulArray[startDataIndex + i] = cluster;
			for (int k = 0; k < amountOfPeaks - 1; k++) {
				if (i != k && helpfulArray[startDataIndex + k] == 0) {
					if (distance(data[startDataIndex + i], intervals[startDataIndex + i], data[startDataIndex + k], intervals[startDataIndex + k]) < eps) {
						helpfulArray[startDataIndex + k] = cluster;
						helpfulArray[startDataIndex + amountOfPeaks + k] = k;
						++NumNeibor;
					}
				}
			}
		}

	return cluster - 1;
}



// ---------------------------------
// --- Ãëîáàëüíàÿ ôóíêöèÿ DBSCAN ---
// ---------------------------------

__global__ void dbscanCUDA(double* data, const int sizeOfBlock, const int amountOfBlocks,
	const int* amountOfPeaks, double* intervals, double* helpfulArray,
	const double eps, int* outData)
{
	// --- Âû÷èñëÿåì èíäåêñ ïîòîêà, â êîòîðîì íàõîäèìñÿ â äàíûé ìîìåíò ---
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= amountOfBlocks)		// Åñëè ñóùåñòâóåò ïîòîê ñ áîëüøèì èíäåêñîì, ÷åì òðåáóåòñÿ - ñðàçó çàâåðøàåì åãî
		return;

	// --- Åñëè íà ïðåäûäóùèõ ýòàïàõ ñèñòåìó óæå îòìåòèëè êàê "íåïðèãîäíóþ", òî ïðîïóñêàåì åå ---
	if (amountOfPeaks[idx] == -1)
	{
		outData[idx] = 0;
		return;
	}

	// --- Ïðèìåíÿåì àëãîðèòì dbscan ê êàæäîé ñèñòåìå
	outData[idx] = dbscan(data, intervals, helpfulArray, idx * sizeOfBlock, amountOfPeaks[idx], sizeOfBlock, idx, eps, outData);
}



// --------------------
// --- ßäðî äëÿ LLE ---
// --------------------
__global__ void LLEKernelCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			resultArray)
{
	extern __shared__ double s[];
	double* x = s + threadIdx.x * amountOfInitialConditions;
	double* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	//printf("%f %f %f %f\n", localValues[0], localValues[1], localValues[2], localValues[3]);

	double zPower = 0;
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		double tempData = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l)
			tempData += (x[l] - y[l]) * (x[l] - y[l]);
		tempData = sqrt(tempData) / eps;

		result += log(tempData);
		
		if (tempData != 0)
			tempData = (1 / tempData);

		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (double)(x[j] - ((x[j] - y[j]) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}



// -------------------------
// --- ßäðî äëÿ LLE (IC) ---
// -------------------------
__global__ void LLEKernelICCUDA(
	const int		nPts,
	const int		nPtsLimiter,
	const double	NT,
	const double	tMax,
	const int		sizeOfBlock,
	const int		amountOfCalculatedPoints,
	const int		amountOfPointsForSkip,
	const int		dimension,
	double*			ranges,
	const double	h,
	const double	eps,
	int*			indicesOfMutVars,
	double*			initialConditions,
	const int		amountOfInitialConditions,
	const double*	values,
	const int		amountOfValues,
	const int		amountOfIterations,
	const int		preScaller,
	const int		writableVar,
	const double	maxValue,
	double*			resultArray)
{
	extern __shared__ double s[];
	double* x = s + threadIdx.x * amountOfInitialConditions;
	double* y = s + (blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* z = s + (2 * blockDim.x + threadIdx.x) * amountOfInitialConditions;
	double* localValues = s + (3 * blockDim.x * amountOfInitialConditions) + (threadIdx.x * amountOfValues);

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
		x[i] = initialConditions[i];

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		x[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	//printf("%f %f %f %f\n", localValues[0], localValues[1], localValues[2], localValues[3]);

	double zPower = 0;
	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		// z[i] = sinf(0.2171828 * (i + 1) + idx + (0.2171828 + i * idx)) * 0.5;
		z[i] = 0.5 * (sinf(idx * (i * idx + 1) + 1));
		zPower += z[i] * z[i];
	}

	zPower = sqrt(zPower);

	for (int i = 0; i < amountOfInitialConditions; i++)
	{
		z[i] /= zPower;
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating

	for (int i = 0; i < amountOfInitialConditions; ++i) {
		y[i] = z[i] * eps + x[i];
	}

	double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error;*/ }

		flag = loopCalculateDiscreteModel(y, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { resultArray[idx] = 0; result;/* goto Error; */ }

		double tempData = 0;

		for (int l = 0; l < amountOfInitialConditions; ++l)
			tempData += (x[l] - y[l]) * (x[l] - y[l]);
		tempData = sqrt(tempData) / eps;

		result += log(tempData);

		if (tempData != 0)
			tempData = (1 / tempData);

		for (int j = 0; j < amountOfInitialConditions; ++j) {
			y[j] = (double)(x[j] - ((x[j] - y[j]) * tempData));
		}
	}

	resultArray[idx] = result / tMax;
}



//find projection operation (ab)
__device__ __host__ void projectionOperator(double* a, double* b, double* minuend, int amountOfValues)
{
	double numerator = 0;
	double denominator = 0;
	for (int i = 0; i < amountOfValues; ++i)
	{
		numerator += a[i] * b[i];
		denominator += b[i] * b[i];
	}

	double fraction = denominator == 0 ? 0 : numerator / denominator;

	for (int i = 0; i < amountOfValues; ++i)
		minuend[i] -= fraction * b[i];
}



__device__ __host__ void gramSchmidtProcess(double* a, double* b, int amountOfVectorsAndValuesInVector, double* denominators=nullptr/*They are is equale for our task*/)
{
	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[j + i * amountOfVectorsAndValuesInVector] = a[j + i * amountOfVectorsAndValuesInVector];

		for (int j = 0; j < i; ++j)
			projectionOperator(a + i * amountOfVectorsAndValuesInVector,
				b + j * amountOfVectorsAndValuesInVector,
				b + i * amountOfVectorsAndValuesInVector,
				amountOfVectorsAndValuesInVector);
	}

	for (int i = 0; i < amountOfVectorsAndValuesInVector; ++i)
	{
		double denominator = 0;
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			denominator += b[i * amountOfVectorsAndValuesInVector + j] * b[i * amountOfVectorsAndValuesInVector + j];
		denominator = sqrt(denominator);
		for (int j = 0; j < amountOfVectorsAndValuesInVector; ++j)
			b[i * amountOfVectorsAndValuesInVector + j] = denominator == 0 ? 0 : b[i * amountOfVectorsAndValuesInVector + j] / denominator;

		if (denominators != nullptr)
			denominators[i] = denominator;
	}
}



__global__ void LSKernelCUDA(
	const int nPts,
	const int nPtsLimiter,
	const double NT,
	const double tMax,
	const int sizeOfBlock,
	const int amountOfCalculatedPoints,
	const int amountOfPointsForSkip,
	const int dimension,
	double* ranges,
	const double h,
	const double eps,
	int* indicesOfMutVars,
	double* initialConditions,
	const int amountOfInitialConditions,
	const double* values,
	const int amountOfValues,
	const int amountOfIterations,
	const int preScaller,
	const int writableVar,
	const double maxValue,
	double* resultArray)
{
	extern __shared__ double s[];

	unsigned long long buferForMem = 0;
	double* x = s + threadIdx.x * amountOfInitialConditions;

	buferForMem += blockDim.x * amountOfInitialConditions;
	double* y = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	double* z = s + buferForMem + amountOfInitialConditions * amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions * amountOfInitialConditions;
	double* localValues = s + buferForMem + amountOfValues * threadIdx.x;

	buferForMem += blockDim.x * amountOfValues;
	double* result = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	buferForMem += blockDim.x * amountOfInitialConditions;
	double* denominators = s + buferForMem + amountOfInitialConditions * threadIdx.x;

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	size_t amountOfNTPoints = NT / h;

	if (idx >= nPtsLimiter)
		return;

	for (int i = 0; i < amountOfInitialConditions; ++i)
	{
		x[i] = initialConditions[i];
		result[i] = 0;
		denominators[i] = 0;
	}

	for (int i = 0; i < amountOfValues; ++i)
		localValues[i] = values[i];

	for (int i = 0; i < dimension; ++i)
		localValues[indicesOfMutVars[i]] = getValueByIdx(amountOfCalculatedPoints + idx,
			nPts, ranges[i * 2], ranges[i * 2 + 1], i);

	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		double zPower = 0;
		for (int i = 0; i < amountOfInitialConditions; ++i)
		{
			z[j * amountOfInitialConditions + i] = sinf(0.2171828 * (i + 1) * (j + 1) + idx + (0.2171828 + i * j * idx)) * 0.5;//0.5 * (sinf(idx * ((1 + i + j) * idx + 1) + 1));	// 0.2171828 change to z[i] = rand(0, 1) - 0.5;
			zPower += z[j * amountOfInitialConditions + i] * z[j * amountOfInitialConditions + i];
		}

		zPower = sqrt(zPower);

		for (int i = 0; i < amountOfInitialConditions; i++)
		{
			z[j * amountOfInitialConditions + i] /= zPower;
		}
	}


	loopCalculateDiscreteModel(x, localValues, h, amountOfPointsForSkip,
		amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);

	//Calculating


	gramSchmidtProcess(z, y, amountOfInitialConditions);


	for (int j = 0; j < amountOfInitialConditions; ++j)
	{
		for (int i = 0; i < amountOfInitialConditions; ++i) {
			y[j * amountOfInitialConditions + i] = y[j * amountOfInitialConditions + i] * eps + x[i];
		}
	}

	//double result = 0;

	for (int i = 0; i < sizeOfBlock; ++i)
	{
		bool flag = loopCalculateDiscreteModel(x, localValues, h, amountOfNTPoints,
			amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
		if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m ) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error;*/ }

		for (int j = 0; j < amountOfInitialConditions; ++j)
		{
			flag = loopCalculateDiscreteModel(y + j * amountOfInitialConditions, localValues, h, amountOfNTPoints,
				amountOfInitialConditions, 1, 0, maxValue, nullptr, idx * sizeOfBlock);
			if (!flag) { for (int m = 0; m < amountOfInitialConditions; ++m) resultArray[idx * amountOfInitialConditions + m] = 0;/* goto Error; */ }
		}

		//I'M STOPPED HERE!!!!!!!!!!!!

		//__syncthreads();

		//NORMALIZTION??????????
		// 
		for (int k = 0; k < amountOfInitialConditions; ++k)
			for (int l = 0; l < amountOfInitialConditions; ++l)
				y[k * amountOfInitialConditions + l] = y[k * amountOfInitialConditions + l] - x[l];

		gramSchmidtProcess(y, z, amountOfInitialConditions, denominators);

		//denominator[amountOfInitialConditions];

		for (int k = 0; k < amountOfInitialConditions; ++k)
		{
			result[k] += log(denominators[k] / eps);

			for (int j = 0; j < amountOfInitialConditions; ++j) {
				y[k * amountOfInitialConditions + j] = (double)(x[j] + z[k * amountOfInitialConditions + j] * eps);
			}
		}
	}

	for (int i = 0; i < amountOfInitialConditions; ++i)
		resultArray[idx * amountOfInitialConditions + i] = result[i] / tMax;
}
