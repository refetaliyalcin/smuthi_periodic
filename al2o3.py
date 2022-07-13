import numpy as np
from scipy.interpolate import interp1d


def al2o3_n(wl):
           
    data = [[302.4, 1.812],
            [310, 1.81],
            [317.9, 1.807],
            [326.3, 1.804],
            [351, 1.801],
            [344.4, 1.799],
            [354.3, 1.796],
            [364.7, 1.794],
            [375.8, 1.791],
            [387.5, 1.789],
            [400, 1.787],
            [413.3, 1.784],
            [427.6, 1.782],
            [442.9, 1.78],
            [459.3, 1.779],
            [476.9, 1.777],
            [496, 1.775],
            [516.7, 1.773],
            [539.1, 1.771],
            [563.6, 1.77],
            [590.5, 1.768],
            [620, 1.767],
            [652.6, 1.765],
            [688.9, 1.764],
            [729.4, 1.762],
            [775, 1.761],
            [826.7, 1.759],
            [953.8, 1.757]]
            
            
    mean_data = np.array(data)
    set_interp = interp1d(mean_data[:, 0], mean_data[:, 1], kind='linear')
    return set_interp(wl)


def ag_n(wl):
    data = [[187.9, 1.07],
            [191.6, 1.1],
            [195.3, 1.12],
            [199.3, 1.14],
            [203.3, 1.15],
            [207.3, 1.18],
            [211.9, 1.2],
            [216.4, 1.22],
            [221.4, 1.25],
            [226.2, 1.26],
            [231.3, 1.28],
            [237.1, 1.28],
            [242.6, 1.3],
            [249, 1.31],
            [255.1, 1.33],
            [261.6, 1.35],
            [268.9, 1.38],
            [276.1, 1.41],
            [284.4, 1.41],
            [292.4, 1.39],
            [300.9, 1.34],
            [310.7, 1.13],
            [320.4, 0.81],
            [331.5, 0.17],
            [342.5, 0.14],
            [354.2, 0.1],
            [367.9, 0.07],
            [381.5, 0.05],
            [397.4, 0.05],
            [413.3, 0.05],
            [430.5, 0.04],
            [450.9, 0.04],
            [471.4, 0.05],
            [495.9, 0.05],
            [520.9, 0.05],
            [548.6, 0.06],
            [582.1, 0.05],
            [616.8, 0.06],
            [659.5, 0.05],
            [704.5, 0.04],
            [756, 0.03],
            [821.1, 0.04],
            [892, 0.04],
            [984, 0.04],
            [1088, 0.04],
            [1216, 0.09],
            [1393, 0.13],
            [1610, 0.15],
            [1937, 0.24]]
    mean_data = np.array(data)
    set_interp = interp1d(mean_data[:, 0], mean_data[:, 1], kind='linear')
    return set_interp(wl)


def ag_k(wl):
    data = [[187.9, 1.212],
            [191.6, 1.232],
            [195.3, 1.255],
            [199.3, 1.277],
            [203.3, 1.296],
            [207.3, 1.312],
            [211.9, 1.325],
            [216.4, 1.336],
            [221.4, 1.342],
            [226.2, 1.344],
            [231.3, 1.357],
            [237.1, 1.367],
            [242.6, 1.378],
            [249, 1.389],
            [255.1, 1.393],
            [261.6, 1.387],
            [268.9, 1.372],
            [276.1, 1.331],
            [284.4, 1.264],
            [292.4, 1.161],
            [300.9, 0.964],
            [310.7, 0.616],
            [320.4, 0.392],
            [331.5, 0.829],
            [342.5, 1.142],
            [354.2, 1.419],
            [367.9, 1.657],
            [381.5, 1.864],
            [397.4, 2.07],
            [413.3, 2.275],
            [430.5, 2.462],
            [450.9, 2.657],
            [471.4, 2.869],
            [495.9, 3.093],
            [520.9, 3.324],
            [548.6, 3.586],
            [582.1, 3.858],
            [616.8, 4.152],
            [659.5, 4.483],
            [704.5, 4.838],
            [756, 5.242],
            [821.1, 5.727],
            [892, 6.312],
            [984, 6.992],
            [1088, 7.795],
            [1216, 8.828],
            [1393, 10.1],
            [1610, 11.85],
            [1937, 14.08]]
    mean_data = np.array(data)
    set_interp = interp1d(mean_data[:, 0], mean_data[:, 1], kind='linear')
    return set_interp(wl)


def sio2_n(wl):
    data = [	[	284.031413612565	,	1.49350649350649	]	,
                    [	357.329842931937	,	1.47727272727273	]	,
                    [	606.020942408377	,	1.46103896103896	]	,
                    [	1000	,	1.44886363636364	]	,
                    [	2610.45751633987	,	1.43089430894309	]	,
                    [	3562.09150326797	,	1.39837398373984	]	,
                    [	5007.8431372549	,	1.33983739837398	]	,
                    [	5886.27450980392	,	1.28130081300813	]	,
                    [	6362.09150326797	,	1.22926829268293	]	,
                    [	6636.60130718954	,	1.18373983739837	]	,
                    [	6966.01307189543	,	1.10569105691057	]	,
                    [	7386.92810457516	,	0.96260162601626	]	,
                    [	7643.13725490196	,	0.80650406504065	]	,
                    [	7935.9477124183	,	0.552845528455285	]	,
                    [	8082.35294117647	,	0.370731707317073	]	,
                    [	8411.76470588235	,	0.442276422764228	]	,
                    [	8924.18300653595	,	0.331707317073171	]	,
                    [	9216.99346405229	,	2.01626016260163	]	,
                    [	9345.09803921569	,	2.31544715447154	]	,
                    [	9436.60130718954	,	2.53658536585366	]	,
                    [	9491.50326797386	,	2.65365853658537	]	,
                    [	9583.00653594771	,	2.78373983739837	]	,
                    [	9656.2091503268	,	2.86829268292683	]	,
                    [	9766.01307189543	,	2.86178861788618	]	,
                    [	9949.01960784314	,	2.63414634146341	]	,
                    [	10095.4248366013	,	2.3609756097561	]	,
                    [	10260.1307189542	,	2.21138211382114	]	,
                    [	10845.7516339869	,	1.97723577235772	]	,
                    [	11230.0653594771	,	1.84715447154472	]	,
                    [	11669.2810457516	,	1.73008130081301	]	,
                    [	11925.4901960784	,	1.6650406504065	]	,
                    [	12675.8169934641	,	1.84715447154472	]	,
                    [	13078.431372549	,	1.92520325203252	]	,
                    [	13426.1437908497	,	1.89918699186992	]	,
                    [	13773.8562091503	,	1.83414634146341	]	,
                    [	14158.1699346405	,	1.75609756097561	]	,
                    [	15000	,	1.58650406504065	]	,
                    [	16208.5308056872	,	1.470703125	]	,
                    [	17819.9052132701	,	1.271484375	]	,
                    [	18759.8736176935	,	1.078125	]	,
                    [	19296.9984202212	,	0.849609375	]	,
                    [	19834.1232227488	,	0.697265625	]	,
                    [	20371.2480252765	,	0.5859375	]	,
                    [	20774.0916271722	,	0.8671875	]	,
                    [	21176.9352290679	,	1.76953125	]	,
                    [	21714.0600315956	,	2.572265625	]	,
                    [	21982.6224328594	,	2.70703125	]	,
                    [	22922.5908372828	,	2.818359375	]	,
                    [	23459.7156398104	,	2.8828125	]	,
                    [	24265.4028436019	,	2.77734375	]	,
                    [	24936.8088467615	,	2.642578125	]	,
                    [	25473.9336492891	,	2.51953125	]	,
                    [	26279.6208530806	,	2.396484375	]	,
                    [	26816.7456556082	,	2.349609375	]	,
                    [	27890.9952606635	,	2.291015625	]	,
                    [	29233.8072669826	,	2.244140625	]	,
                    [	30710.9004739336	,	2.21484375	]	,
                    [	32187.9936808847	,	2.19140625	]	]
    mean_data = np.array(data)
    set_interp = interp1d(mean_data[:, 0], mean_data[:, 1], kind='linear')
    return set_interp(wl)


def sio2_k(wl):
    data = [	[	30.0653594771242	,	0.071292951545487	]	,
                    [	37.9084967320261	,	0.133489783491451	]	,
                    [	49.6732026143791	,	0.28523126801432	]	,
                    [	56.2091503267974	,	0.438103841136116	]	,
                    [	74.5098039215686	,	0.767899985694281	]	,
                    [	96.7320261437909	,	0.767899985694281	]	,
                    [	113.725490196078	,	0.767899985694281	]	,
                    [	121.56862745098	,	0.589670388029277	]	,
                    [	129.411764705882	,	0.062473944110854	]	,
                    [	142.483660130719	,	0.008908832396093	]	,
                    [	150.326797385621	,	0.000913215925902	]	,
                    [	154.248366013072	,	0.000114116296898	]	,
                    [	158.169934640523	,	2.05E-05	]	,
                    [	163.398692810458	,	4.35E-06	]	,
                    [	167.320261437908	,	1.51E-06	]	,
                    [	188.235294117647	,	1.16E-06	]	,
                    [	207.843137254902	,	1.16E-06	]	,
                    [	224.83660130719	,	5.43E-07	]	,
                    [	241.830065359477	,	9.84E-07	]	,
                    [	269.281045751634	,	6.35E-08	]	,
                    [	309.803921568627	,	7.01E-08	]	,
                    [	359.477124183007	,	1.36E-07	]	,
                    [	380.392156862745	,	1.04E-07	]	,
                    [	398.692810457516	,	6.78E-08	]	,
                    [	481.045751633987	,	7.74E-08	]	,
                    [	562.091503267974	,	7.74E-08	]	,
                    [	622.222222222222	,	7.49E-08	]	,
                    [	658.823529411765	,	1.08E-07	]	,
                    [	679.738562091503	,	9.44E-08	]	,
                    [	803.921568627451	,	1.23E-07	]	,
                    [	843.137254901961	,	1.11E-07	]	,
                    [	881.045751633987	,	4.88E-08	]	,
                    [	922.875816993464	,	9.44E-08	]	,
                    [	960.78431372549	,	9.76E-08	]	,
                    [	1000	,	8.55E-08	]	,
                    [	1219.32114882507	,	4.00E-08	]	,
                    [	1420.36553524804	,	7.74E-08	]	,
                    [	1566.57963446475	,	4.72E-08	]	,
                    [	1932.11488250653	,	4.17E-07	]	,
                    [	2571.80156657963	,	2.32E-06	]	,
                    [	3321.14882506527	,	4.49E-06	]	,
                    [	3796.34464751958	,	4.84E-05	]	,
                    [	4472.58485639687	,	0.000307224040918	]	,
                    [	4765.01305483029	,	0.000471883862357	]	,
                    [	5002.61096605744	,	0.005082684940067	]	,
                    [	6483.02872062663	,	0.006618941313655	]	,
                    [	7104.43864229765	,	0.010166429627278	]	,
                    [	7488.25065274151	,	0.024789355346435	]	,
                    [	7671.0182767624	,	0.064570751703294	]	,
                    [	7926.89295039165	,	0.137970089569798	]	,
                    [	9023.49869451697	,	2.69220128298041	]	,
                    [	9754.56919060052	,	0.71883766429437	]	,
                    [	10101.8276762402	,	0.304698957090351	]	,
                    [	10869.4516971279	,	0.18570226648111	]	,
                    [	11308.0939947781	,	0.152333597657718	]	,
                    [	12039.1644908616	,	0.275968946102348	]	,
                    [	12605.7441253264	,	0.371443252152732	]	,
                    [	13154.046997389	,	0.226380340952145	]	,
                    [	13793.7336814621	,	0.078714972429348	]	,
                    [	14177.545691906	,	0.04797374078899	]	,
                    [	14469.9738903394	,	0.043450305221094	]	,
                    [	15000	,	0.054745856471882	]	,
                    [	15670.3470031546	,	0.080595306538209	]	,
                    [	16206.6246056782	,	0.122988967837561	]	,
                    [	16742.9022082019	,	0.168862595693521	]	,
                    [	17413.2492113565	,	0.219915689349749	]	,
                    [	18485.8044164038	,	0.305956480048725	]	,
                    [	19424.2902208202	,	0.466891600772887	]	,
                    [	19960.5678233438	,	0.791883999516008	]	,
                    [	20362.7760252366	,	1.3609532662757	]	,
                    [	20899.0536277603	,	1.86857492343883	]	,
                    [	21435.3312302839	,	2.33897110450336	]	,
                    [	22507.8864353312	,	1.32547315675008	]	,
                    [	23178.2334384858	,	0.891839712197086	]	,
                    [	23848.5804416404	,	0.584427174658512	]	,
                    [	24652.9968454259	,	0.326843870030386	]	,
                    [	25189.2744479495	,	0.241217703139677	]	,
                    [	25591.4826498423	,	0.208598731667757	]	,
                    [	26127.7602523659	,	0.190177309853725	]	,
                    [	27468.4542586751	,	0.173382689789336	]	,
                    [	30015.7728706625	,	0.1441118883046	]	,
                    [	38462.1451104101	,	0.067879707425743	]	,
                    [	49858.0441640379	,	0.027286413911149	]	]
    mean_data = np.array(data)
    set_interp = interp1d(mean_data[:, 0], mean_data[:, 1], kind='linear')
    return set_interp(wl)