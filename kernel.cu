// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cuda_runtime.h>

//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define SEED 124

#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }

template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

enum Mode
{
    ModeProcedural,
    ModeToeplitz,
    ModeMax,
};

// Convolution kernel
const float s_Kernel[] = {
    1, 0, -1,
    2, 0, -2,
    1, 0, -1};

// Array of input spikes
const unsigned int s_Spikes[] = {
      65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,
      76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,
      87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,
      98,   99,  100,  101,  102,  103,  104,  105,  106,  107,  108,
     109,  110,  129,  130,  131,  132,  133,  134,  135,  136,  137,
     138,  139,  140,  141,  142,  143,  144,  145,  146,  147,  148,
     149,  150,  151,  152,  153,  154,  155,  156,  157,  158,  159,
     160,  161,  162,  163,  164,  165,  166,  167,  168,  169,  170,
     171,  172,  173,  174,  193,  194,  195,  196,  197,  198,  199,
     200,  201,  202,  203,  204,  205,  206,  207,  208,  209,  210,
     211,  212,  213,  214,  215,  257,  258,  259,  260,  261,  262,
     263,  264,  265,  266,  267,  268,  269,  270,  271,  272,  273,
     274,  275,  276,  277,  278,  279,  321,  322,  323,  324,  325,
     326,  327,  328,  329,  330,  331,  332,  333,  334,  335,  336,
     337,  338,  339,  340,  341,  342,  343,  385,  386,  387,  388,
     389,  390,  391,  392,  393,  394,  395,  396,  397,  398,  399,
     400,  401,  402,  403,  404,  405,  406,  407,  449,  450,  451,
     452,  453,  454,  455,  456,  457,  458,  459,  460,  461,  462,
     463,  464,  465,  466,  467,  468,  469,  470,  471,  507,  508,
     513,  514,  515,  516,  517,  518,  519,  520,  521,  522,  523,
     524,  525,  526,  527,  528,  529,  530,  531,  532,  533,  534,
     535,  568,  569,  570,  571,  577,  578,  579,  580,  581,  582,
     583,  584,  585,  586,  587,  588,  589,  590,  591,  592,  593,
     594,  595,  596,  597,  598,  599,  629,  630,  631,  632,  633,
     634,  641,  642,  643,  644,  645,  646,  647,  648,  649,  650,
     651,  652,  653,  654,  655,  656,  657,  658,  659,  660,  661,
     662,  663,  690,  691,  692,  693,  694,  695,  696,  697,  705,
     706,  707,  708,  709,  710,  711,  712,  713,  714,  715,  716,
     717,  718,  719,  720,  721,  722,  723,  724,  725,  726,  727,
     750,  751,  752,  753,  754,  755,  756,  757,  758,  759,  760,
     769,  770,  771,  772,  773,  774,  775,  776,  777,  778,  779,
     780,  781,  782,  783,  784,  785,  786,  787,  788,  789,  790,
     791,  810,  811,  812,  813,  814,  815,  816,  817,  818,  819,
     820,  821,  822,  823,  833,  834,  835,  836,  837,  838,  839,
     840,  841,  842,  843,  844,  845,  846,  847,  848,  849,  850,
     851,  852,  853,  854,  855,  871,  872,  873,  874,  875,  876,
     877,  878,  879,  880,  881,  882,  883,  884,  885,  886,  897,
     898,  899,  900,  901,  902,  903,  904,  905,  906,  907,  908,
     909,  910,  911,  912,  913,  914,  915,  916,  917,  918,  919,
     932,  933,  934,  935,  936,  937,  938,  939,  940,  941,  942,
     943,  944,  945,  946,  947,  948,  949,  992,  993,  994,  995,
     996,  997,  998,  999, 1000, 1001, 1002, 1003, 1004, 1005, 1006,
    1007, 1008, 1009, 1010, 1011, 1012, 1053, 1054, 1055, 1056, 1057,
    1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068,
    1069, 1070, 1071, 1072, 1073, 1074, 1075, 1114, 1115, 1116, 1117,
    1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
    1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1174,
    1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185,
    1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196,
    1197, 1198, 1199, 1200, 1201, 1217, 1218, 1219, 1220, 1221, 1222,
    1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1235,
    1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246,
    1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257,
    1258, 1259, 1260, 1261, 1262, 1263, 1264, 1281, 1282, 1283, 1284,
    1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295,
    1296, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308,
    1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319,
    1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1345, 1346,
    1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357,
    1358, 1359, 1360, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373,
    1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384,
    1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1409, 1410,
    1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421,
    1422, 1423, 1424, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440,
    1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451,
    1452, 1453, 1454, 1455, 1456, 1457, 1458, 1473, 1474, 1475, 1476,
    1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487,
    1488, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510,
    1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521,
    1522, 1523, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545,
    1546, 1547, 1548, 1549, 1550, 1551, 1552, 1568, 1569, 1570, 1571,
    1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582,
    1583, 1584, 1585, 1586, 1587, 1588, 1601, 1602, 1603, 1604, 1605,
    1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616,
    1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646,
    1647, 1648, 1649, 1650, 1651, 1652, 1653, 1665, 1666, 1667, 1668,
    1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679,
    1680, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712,
    1713, 1714, 1715, 1716, 1717, 1718, 1729, 1730, 1731, 1732, 1733,
    1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744,
    1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780,
    1781, 1782, 1783, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800,
    1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1837, 1838, 1839,
    1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1857, 1858,
    1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869,
    1870, 1871, 1872, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911,
    1912, 1913, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929,
    1930, 1931, 1932, 1933, 1934, 1935, 1936, 1972, 1973, 1974, 1975,
    1976, 1977, 1978, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2039, 2040, 2041,
    2042, 2043, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057,
    2058, 2059, 2060, 2061, 2062, 2063, 2064, 2106, 2107, 2108, 2267,
    2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278,
    2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2305,
    2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316,
    2317, 2318, 2319, 2320, 2321, 2322, 2323, 2331, 2332, 2333, 2334,
    2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345,
    2346, 2347, 2348, 2349, 2350, 2351, 2352, 2370, 2371, 2372, 2373,
    2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384,
    2385, 2386, 2387, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402,
    2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413,
    2414, 2415, 2416, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442,
    2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2459, 2460,
    2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471,
    2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2500, 2501,
    2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512,
    2513, 2514, 2515, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530,
    2531, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541,
    2542, 2543, 2544, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572,
    2573, 2574, 2575, 2576, 2577, 2578, 2579, 2587, 2588, 2589, 2590,
    2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601,
    2602, 2603, 2604, 2605, 2606, 2607, 2608, 2630, 2631, 2632, 2633,
    2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2651,
    2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662,
    2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2695,
    2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706,
    2707, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724,
    2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735,
    2736, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769,
    2770, 2771, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787,
    2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798,
    2799, 2800, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833,
    2834, 2835, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851,
    2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862,
    2863, 2864, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898,
    2899, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916,
    2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927,
    2928, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2971,
    2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982,
    2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 3020,
    3021, 3022, 3023, 3024, 3025, 3026, 3027, 3035, 3036, 3037, 3038,
    3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049,
    3050, 3051, 3052, 3053, 3054, 3055, 3056, 3085, 3086, 3087, 3088,
    3089, 3090, 3091, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106,
    3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117,
    3118, 3119, 3120, 3150, 3151, 3152, 3153, 3154, 3155, 3215, 3216,
    3217, 3218, 3219, 3280, 3281, 3282, 3283, 3345, 3346, 3347, 3410,
    3411, 3475, 3547, 3548, 3549, 3550, 3551, 3608, 3609, 3610, 3611,
    3612, 3613, 3614, 3615, 3616, 3617, 3618, 3669, 3670, 3671, 3672,
    3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683,
    3684, 3685, 3686, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737,
    3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748,
    3749, 3750, 3751, 3752, 3753, 3790, 3791, 3792, 3793, 3794, 3795,
    3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806,
    3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817,
    3818, 3819, 3820, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858,
    3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869,
    3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880,
    3881, 3882, 3883, 3884, 3885, 3886, 3887, 3912, 3913, 3914, 3915,
    3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926,
    3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937,
    3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948,
    3949, 3950, 3951, 3952, 3953, 3954, 3973, 3974, 3975, 3976, 3977,
    3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988,
    3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999,
    4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010,
    4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021,
    4022, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044,
    4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055,
    4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066,
    4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077,
    4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088};

const char *const s_ModeNames[] = {
    "Procedural",
    "Toeplitz"};

//-----------------------------------------------------------------------------
// Kernels
//-----------------------------------------------------------------------------
template<int ConvKH, int ConvKW,
         int ConvIW, int ConvIC,
         int ConvOH, int ConvOW, int ConvOC>
__global__ void procedural(unsigned int numInSpikes, const unsigned int *d_inSpikes,
                           const float *d_kernel, float *d_outCurrents)
{
    const unsigned int spike = threadIdx.x + (blockIdx.x * blockDim.x);

    if (spike < numInSpikes) {
        const unsigned int preInd = d_inSpikes[spike];
        const int inRow = (preInd / ConvIC) / ConvIW;
        const int inCol = (preInd / ConvIC) % ConvIW;
        const int inChan = preInd % ConvIC;
        const int minOutRow = min(ConvOH, max(0, 1 + (inRow - ConvKH)));
        const int maxOutRow = min(ConvOH, max(0, 1 + inRow));
        const int minOutCol = min(ConvOW, max(0, 1 + (inCol - ConvKW)));
        const int maxOutCol = min(ConvOW, max(0, 1 + inCol));
        for(int outRow = minOutRow; outRow < maxOutRow; outRow++) {
            const int kernRow = inRow - outRow;
            for(int outCol = minOutCol; outCol < maxOutCol; outCol++) {
                const int kernCol = inCol - outCol;
                for(int outChan = 0; outChan < ConvOC; outChan++) {
                    const int idPost = ((outRow * ConvOW * ConvOC) +
                                        (outCol * ConvOC) +
                                        outChan);
                    const unsigned int kernelInd = (kernRow * ConvKW * ConvIC * ConvOC) + (kernCol * ConvIC * ConvOC) + (inChan * ConvOC) + outChan;
                    atomicAdd(&d_outCurrents[idPost], d_kernel[kernelInd]);
                }
            }
        }
    }
}

template<int ConvK, int ConvI, int ConvIC, int ConvO, int ConvOC>
__global__ void toeplitz(unsigned int numInSpikes, const unsigned int *d_inSpikes,
                         const float *d_kernel, float *d_outCurrents)
{
    extern __shared__ unsigned int s_buffer[];
    unsigned int *s_spike = &s_buffer[0];

    const int id = threadIdx.x + (blockIdx.x * blockDim.x);

    // Split id into kernel row, column and output channel
    const int kernRow = (id / ConvOC) / ConvK;
    const int kernCol = (id / ConvOC) % ConvK;
    const int kernOutChan = id % ConvOC;
    
    // From these, calculate partial (without input channel) kernel index
    const int kernelInd = (kernRow * ConvK * ConvIC * ConvOC) + (kernCol * ConvIC * ConvOC) + kernOutChan;
    
    const int postInd = ((outRow * ConvO * ConvOC) +
                                        (outCol * ConvOC) +
                                        kernOutChan)
    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikeBlocks = (numInSpikes + blockDim.x - 1) / blockDim.x;

    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numInSpikes - 1) % blockDim.x) + 1 : blockDim.x;
     
        __syncthreads();
            
        // Use first threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            s_spike[threadIdx.x] = d_inSpikes[(b * blockDim.x) + threadIdx.x];
        }

        __syncthreads();

        // If there is a kernel entry for this thread to process
        if(id < (ConvO * ConvO * ConvOC)) {
            // Loop through spikes in block
            for(unsigned int s = 0; s < numSpikesInBlock; s++) {
                // Split pre into row, column and channel
                // **NOTE** this COULD be done once and written to shared memory
                const int preRow = (s_spike[s] / ConvIC) / ConvI;
                const int preCol = (s_spike[s] / ConvIC) % ConvI;
                const int preChan = s_Spikes[s] % ConvIC;

                // If we haven't gone off edge of output
                const int i = preRow + kernRow;
                if(i < ConvO && kernCol < (ConvO - preCol)) {
                    const int startOut = (i * ConvO) + preCol;
                    
                    // Read kernel value
                    // **NOTE** if we were only processing a single input channel this could be lifted right out
                    const float kernelVal = d_kernel[kernelInd + (preChan * ConvOC)];
                    
                    // Update output (coalesced reading of filter row and no collisions on atomic add)
                    atomicAdd(&d_outCurrents[startOut + kernCol], kernelVal);
                }
            }
        }
        
    }
}


//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
//! Divide two integers, rounding up i.e. effectively taking ceil
template<typename T>
constexpr T ceilDivide(T numerator, T denominator)
{
    return ((numerator + denominator - 1) / denominator);
}
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(unsigned int count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, unsigned int count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, unsigned int count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, count * sizeof(T), cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try
    {
        constexpr int blockSize = 128;
        constexpr int convKH = 3;
        constexpr int convKW = 3;
        constexpr int convIH = 64;
        constexpr int convIW = 64;
        constexpr int convIC = 1;
        constexpr int convOH = 62;
        constexpr int convOW = 62;
        constexpr int convOC = 1;
        constexpr unsigned int numSpikesPerTimestep = 100;

        // Calculate sizes of kernels and neuron populations
        constexpr int numPre = convIH * convIW * convIC;
        constexpr int numPost = convOH * convOW * convOC;
        constexpr int kernelSize = convKW * convKH * convIC * convOC;

        // Count spikes
        constexpr unsigned int numSpikes = sizeof(s_Spikes) / sizeof(unsigned int);
        
        // Calculate required timesteps
        constexpr unsigned int numTimesteps =  ceilDivide(numSpikes, numSpikesPerTimestep);

        // Calculate remaining spikes to process in last timestep
        constexpr unsigned int lastTimestepSpikes = numSpikes - ((numTimesteps - 1) * numSpikesPerTimestep);
        
        // Check filter is correct size
        assert((sizeof(s_Kernel) / sizeof(float)) == kernelSize);

        // Read mode from command line
        Mode mode;
        if(argc < 2) {
            std::cerr << "Expected parameters specifying:" << std::endl;
            std::cerr << "\t Mode (";
            for(int m = 0; m < ModeMax; m++) {
                std::cerr << m << " = " << s_ModeNames[m];
                if(m != (ModeMax - 1)) {
                    std::cerr << ", ";
                }
            }
            std::cerr << ")" << std::endl;
            return EXIT_FAILURE;
        }
        else {
            mode = (Mode)std::stoul(argv[1]);
        }

        std::cout << "Mode:" << s_ModeNames[mode] << std::endl;
    
        CHECK_CUDA_ERRORS(cudaSetDevice(0));

        //------------------------------------------------------------------------
        // Configure fixed-probability connector
        //------------------------------------------------------------------------
        // Create arrays to hold post-synaptic currents
        auto outCurrents = allocateHostDevice<float>(numPost);
        std::fill_n(&outCurrents.first[0], numPost, 0.0f);
        hostToDeviceCopy(outCurrents, numPost);

        // Create device array for kernels and copy in global data
        float *d_kernel = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_kernel, kernelSize * sizeof(float)));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_kernel, s_Kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice));


        // Create device array for spikes and copy in global data
        unsigned int *d_spikes = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_spikes, numSpikes * sizeof(unsigned int)));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_spikes, s_Spikes, numSpikes * sizeof(unsigned int), cudaMemcpyHostToDevice));

        {
            // Loop through time
            for(unsigned int t = 0; t < numTimesteps; t++) {
                const unsigned int numTimestepSpikes = (t == (numTimesteps - 1)) ? lastTimestepSpikes : numSpikesPerTimestep;

                if(mode == ModeProcedural) {
                    // Calculate number of presynaptically parallelised blocks are required to handle poisson spikes
                    constexpr unsigned int numPreSynapseBlocks = ceilDivide(numPre, blockSize);

                    dim3 threads(blockSize, 1);
                    dim3 grid(numPreSynapseBlocks, 1);

                    procedural<convKH, convKW, convIW, convIC, convOH, convOW, convOC><<<grid, threads>>> (
                        numTimestepSpikes, &d_spikes[t * numSpikesPerTimestep],
                        d_kernel, outCurrents.second);
                }
                else if(mode == ModeToeplitz) {
                    assert(convKH == convKW);
                    assert(convIW == convIH);
                    assert(convOW == convOH);
                    assert(convIC == 1);
                    assert(convOC == 1);

                    constexpr unsigned int numPostSynapseBlocks = ceilDivide(convKW * convKH, blockSize);
                    constexpr unsigned int sharedBytes = blockSize * sizeof(unsigned int);
                    
                    dim3 threads(blockSize, 1);
                    dim3 grid(numPostSynapseBlocks, 1);
                    toeplitz<convKH, convIH, convOH><<<grid, threads, sharedBytes>>>(
                        numTimestepSpikes, &d_spikes[t * numSpikesPerTimestep],
                        d_kernel, outCurrents.second);
                }
                CHECK_CUDA_ERRORS(cudaPeekAtLastError());
            }
        }
        deviceToHostCopy(outCurrents, numPost);

        std::ofstream outCurrentsFile("outCurrents" + std::string(s_ModeNames[mode]) + ".bin", std::ios_base::binary);
        outCurrentsFile.write(reinterpret_cast<const char*>(outCurrents.first), sizeof(float) * numPost);
    }
    catch(std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

