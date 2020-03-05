# first model (foo)
devel:
  0.967	0.936	0.951

Predicted   no  yes
Actual
no         954    8
yes         98  939

unindentified:
	1.000	0.897	0.945

Predicted   no   yes
Actual
yes        744  3870

annotated
  0.870	0.886	0.878

Predicted    no   yes
Actual
no         3814   282
yes         792  3305


# trained longer (20, en_core_web_sm)
0.979	0.961	0.970

Predicted   no   yes
Actual
no         917    45
yes         34  1003

unindentified	1.000	0.923	0.960

Predicted   no   yes
Actual
yes        284  4330

annotated	0.862	0.915	0.888

Predicted    no   yes
Actual
no         3303   793
yes         280  3817

# tf: cnn, one layer
eval    0.953   0.977   0.965

Predicted    0     1
Actual
0          912    50
1           24  1013

unindentified   1.000   0.893   0.944

Predicted    0     1
Actual
1          492  4122

annotated       0.832   0.910   0.869

Predicted     0     1
Actual
0          3342   754
1           370  3727

# tf: cnn, multiple layers
eval    0.983   0.959   0.971

Predicted    0    1
Actual
0          945   17
1           42  995


unindentified   1.000   0.911   0.953

Predicted    0     1
Actual
1          412  4202

annotated       0.846   0.906   0.875

Predicted     0     1
Actual
0          3420   676
1           385  3712


# tf: simple lstm
eval    0.901   0.943   0.921
I [2019-09-03 16:04:12,863] [tk_nn_classifier] Confusion matrix:
Predicted    0    1
Actual
0          854  108
1           59  978


unindentified   1.000   0.918   0.957
I [2019-09-03 16:03:04,564] [tk_nn_classifier] Confusion matrix:
Predicted    0     1
Actual
1          377  4237


annotated       0.770   0.916   0.837
I [2019-09-03 16:04:08,662] [tk_nn_classifier] Confusion matrix:
Predicted     0     1
Actual
0          2977  1119
1           343  3754



# tf: multi-layer lstm
difficult to train and restult are not reliable yet


# training on big set:
spacy
unindentified   1.000   0.915   0.956
Predicted   no   yes
Actual
yes        434  4180

annotated       0.993   0.990   0.992
Predicted    no   yes
Actual
no         4077    19
yes          58  4039

us     0.946   0.138   0.241
Predicted    no  yes
Actual
no         9921   79
yes        9092  908

tf_cnn_simple
unindentified	1.000	0.936	0.967
Predicted    0     1
Actual
1          297  4317

annotated	0.972	0.985	0.979
Predicted     0     1
Actual
0          3981   115
1            62  4035

eval 	0.986	0.987	0.987
Predicted    0     1
Actual
0          947    15
1           13  1024

us  	0.922	0.697	0.794
Predicted     0     1
Actual
0          9409   591
1          3026  6974

# retrain: cnn-simple on train_big, test on eval_big
accuracy': 0.8910116, 'auc': 0.8908284, 'loss': 0.28237858, 'global_step': 2900

- unindentified:
label   Prec    Reca     F1
  0     0.000   0.000   0.000
  1     1.000   0.923   0.960

Predicted    0     1
Actual
1          355  4259

- annotated:
label   Prec    Reca     F1
  0     0.919   0.815   0.864
  1     0.834   0.928   0.878
I [2019-11-29 16:43:24,058] [tk_nn_classifier] Confusion matrix:
Predicted     0     1
Actual
0          3338   758
1           294  3803

- eval:
label   Prec    Reca     F1
  0     0.947   0.861   0.902
  1     0.881   0.956   0.917
I [2019-11-29 16:43:44,298] [tk_nn_classifier] Confusion matrix:
Predicted    0    1
Actual
0          828  134
1           46  991


- random:
label   Prec    Reca     F1
  0     0.633   0.216   0.322
  1     0.538   0.879   0.667
I [2019-11-29 16:45:28,663] [tk_nn_classifier] Confusion matrix:
Predicted     0     1
Actual
0          1142  4138
1           663  4810

- us:
label   Prec    Reca     F1
  0     0.649   0.828   0.728
  1     0.763   0.552   0.640
I [2019-11-29 16:47:06,263] [tk_nn_classifier] Confusion matrix:
Predicted     0     1
Actual
0          8284  1716
1          4482  5518

# retrain with uppercase everywhere:
accuracy': 0.9680551, 'auc': 0.9680475, 'loss': 0.11155169, 'global_step': 1300

- unidentified:
label   Prec    Reca     F1
  0     0.000   0.000   0.000
  1     1.000   0.959   0.979
I [2019-11-29 17:03:28,179] [tk_nn_classifier] Confusion matrix:
Predicted    0     1
Actual
1          188  4426


- annotated:
label   Prec    Reca     F1
  0     0.981   0.950   0.965
  1     0.952   0.981   0.966
I [2019-11-29 17:04:44,830] [tk_nn_classifier] Confusion matrix:
Predicted     0     1
Actual
0          3892   204
1            77  4020

- eval:
label   Prec    Reca     F1
  0     0.987   0.974   0.981
  1     0.976   0.988   0.982
I [2019-11-29 17:05:02,664] [tk_nn_classifier] Confusion matrix:
Predicted    0     1
Actual
0          937    25
1           12  1025


- random:
label   Prec    Reca     F1
  0     0.848   0.221   0.351
  1     0.561   0.962   0.709
I [2019-11-29 17:06:32,817] [tk_nn_classifier] Confusion matrix:
Predicted     0     1
Actual
0          1169  4111
1           210  5263

- us:
label   Prec    Reca     F1
  0     0.748   0.937   0.832
  1     0.916   0.685   0.783
I [2019-11-29 17:08:09,777] [tk_nn_classifier] Confusion matrix:
Predicted     0     1
Actual
0          9369   631
1          3154  6846


# all en data:

- baseline
134957 loaded_data/all_data.csv

- v1: deduplicat on text:
133825 loaded_data/all_data.csv

- v2: removed all small docs with len < 500
124991 loaded_data/all_data.csv

- v3: reduce the number of CV to maximum 20 per org per data_set
113406

- v4: exclude random_anno, splitted data into train/eval
random_anno: 1449

annotated.csv                          8220
AT_129_staffing_postings.csv       548
AT_490_direct_postings.csv         3294
BE_1000_direct_postings.csv        8093
BE_385_staffing_postings.csv       3773
CA_1000_direct_postings.csv        13674
CA_597_staffing_postings.csv       8427
DE_1000_direct_postings.csv        6376
DE_618_staffing_postings.csv       3615
ES_210_staffing_postings.csv       1735
ES_807_direct_postings.csv         6501
FR_1000_direct_postings.csv        7348
FR_225_staffing_postings.csv       1816
IT_137_staffing_postings.csv       855
IT_744_direct_postings.csv         5287
NL_1000_direct_postings.csv        8288
NL_765_staffing_postings.csv       6512
uk_directed_employer.csv           4216
uk_staffing_agency.csv             3195
direct_us.csv                      7372
staffing_us.csv                    2762


all_data.csv                           111907
train.csv                              90103  => yes: 30097, no: 59310
eval.csv                               21804
eval_other.csv                         16879
eval_uk.csv                            2935
eval_us.csv                            1990



ratio: 0.9
I [2019-12-27 07:17:17,255] [data_aggre] in total 11038 orgs
I [2019-12-27 07:17:17,262] [data_aggre] splitted orgs into train: 9934, eval: 1104

ratio: 0.8
I [2019-12-27 07:36:02,466] [data_aggre] in total 11038 orgs
I [2019-12-27 07:36:02,472] [data_aggre] splitted orgs into train: 8830, eval: 2208

# design
- all_data => train, evel, random_anno

eval => eval_us, eval_uk, eval_other

test => random_anno

# current_version:

I [2019-12-27 08:25:45,489] [tk_nn_classifier] save result to [res/eval.tsv]
label   Prec    Reca     F1
 no     0.847   0.929   0.887
 yes    0.812   0.646   0.720
I [2019-12-27 08:25:45,584] [tk_nn_classifier] Confusion matrix:
Predicted       no   yes
Actual
no             13645  1036
yes             2456  4484


I [2019-12-27 08:25:45,588] [tk_nn_classifier] process test_set [eval_us]
I [2019-12-27 08:26:00,160] [tk_nn_classifier] save result to [res/eval_us.tsv]
label   Prec    Reca     F1
 no     0.833   0.960   0.892
 yes    0.801   0.453   0.579
I [2019-12-27 08:26:00,178] [tk_nn_classifier] Confusion matrix:
Predicted      no  yes
Actual
no             1404   58
yes             282  234


I [2019-12-27 08:26:00,182] [tk_nn_classifier] process test_set [eval_uk]
I [2019-12-27 08:26:20,110] [tk_nn_classifier] save result to [res/eval_uk.tsv]
label   Prec    Reca     F1
 no     0.990   0.980   0.985
 yes    0.972   0.986   0.979
I [2019-12-27 08:26:20,131] [tk_nn_classifier] Confusion matrix:
Predicted      no   yes
Actual
no             1676    34
yes              17  1165

I [2019-12-27 08:26:20,134] [tk_nn_classifier] process test_set [eval_other]
I [2019-12-27 08:28:11,860] [tk_nn_classifier] save result to [res/eval_other.tsv]
label   Prec    Reca     F1
 no     0.830   0.918   0.872
 yes    0.766   0.589   0.666
I [2019-12-27 08:28:11,929] [tk_nn_classifier] Confusion matrix:
Predicted       no   yes
Actual
no             10565   944
yes             2157  3085

I [2019-12-27 08:28:11,933] [tk_nn_classifier] process test_set [random_anno]
I [2019-12-27 08:28:20,539] [tk_nn_classifier] save result to [res/random_anno.tsv]
label   Prec    Reca     F1
 no     0.449   0.506   0.476
 yes    0.933   0.917   0.925
I [2019-12-27 08:28:20,555] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no            89    87
yes          109  1204

manual evaluation:
- Mortgage Advice Bureau (8 jobs)
 c2393aa009634183beb40e6dcfb33a50: You will start your journey with the business with them providing you a good insight into the business; how they work, what their culture is and how they deal with their customers.
 ddef67862caa40fb9606037d7d09f393: One of our highly respected, market leading & award winning mortgage broker & national estate agency business partners is seeking fully CeMAP (or equivalent) qualified Mortgage Advisers in the ...
- Huntswood (10 jobs)
 ee30052630b342fdb0cbfd3ca5e501f6: Huntswood is a leading provider of outsourcing, resourcing and advisory services. Our aim is to drive better outcomes by combining people, processes and technology to deliver practical solutions to our clients.
- Southampton Education Hampshire SDC (> 10 jobs)
 c643ed4518c94c4d88791dfe8945ff1e: At Hampshire Supply Service we strive on doing the best for our candidates and our schools. Now that we are in direct partnership with Hampshire County Council our supply chain has ex
clusive access to many Hampshire schools, meaning we have plenty of day-to-day supply and long term positions available through us.
- Provide Education
 ccd7d3a5556f435ba5be1500e2f30860: Provide Education are currently recruiting for Cover Supervisors for a variety of day-to-day, short and long term supply roles in several schools in the Chesterfield area.
- HGV Training Network


seems real erros:
- Embrace Financial Services
- Mortgage Advice Bureau (MAB)


# retrain: v1_1024 =>  input length: 1024

parameter searching: CNN_simple
cnn_retrain_v1_1024_0.003: 0.003: devel (5200 steps)
0     0.944   0.763   0.844
1     0.643   0.904   0.751

cnn_retrain_v1_1024_0.002: 0.002: devel (4200 steps)
0     0.932   0.831   0.879
1     0.709   0.873   0.782

cnn_retrain_v1_1024_0.001: 0.001: devel (9100 steps)
  0     0.920   0.886   0.903
  1     0.777   0.838   0.806

parameter searching: CNN_multi
cnn_multi_v1_1024_0.001: 3 layer, 0.001: devel (8100):
0     0.923   0.896   0.909
1     0.793   0.841   0.816

cnn_multi_v2_1024_0.001: max_line: 50 => 500, 3 layer, 0.001 (7300) => accuracy': 0.89054686
0     0.929   0.879   0.903
1     0.770   0.858   0.812

retrained on tk118 (13000):
I [2020-01-02 02:56:03,272] [tk_nn_classifier] process test_set [eval]
I [2020-01-02 02:59:32,077] [tk_nn_classifier] save result to [res/eval.tsv]
label   Prec    Reca     F1
 no     0.909   0.933   0.921
 yes    0.850   0.803   0.826
I [2020-01-02 02:59:32,692] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         13700   981
yes         1369  5571
I [2020-01-02 02:59:32,697] [tk_nn_classifier] process test_set [eval_us]
I [2020-01-02 02:59:52,231] [tk_nn_classifier] save result to [res/eval_us.tsv]
label   Prec    Reca     F1
 no     0.918   0.964   0.941
 yes    0.882   0.756   0.814
I [2020-01-02 02:59:52,334] [tk_nn_classifier] Confusion matrix:
Predicted    no  yes
Actual
no         1410   52
yes         126  390
I [2020-01-02 02:59:52,338] [tk_nn_classifier] process test_set [eval_uk]
I [2020-01-02 03:00:16,946] [tk_nn_classifier] save result to [res/eval_uk.tsv]
label   Prec    Reca     F1
 no     0.966   0.942   0.954
 yes    0.919   0.952   0.935
I [2020-01-02 03:00:17,050] [tk_nn_classifier] Confusion matrix:
Predicted    no   yes
Actual
no         1611    99
yes          57  1125
I [2020-01-02 03:00:17,053] [tk_nn_classifier] process test_set [eval_other]
I [2020-01-02 03:02:56,589] [tk_nn_classifier] save result to [res/eval_other.tsv]
label   Prec    Reca     F1
 no     0.900   0.928   0.914
 yes    0.830   0.774   0.801
I [2020-01-02 03:02:56,919] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         10679   830
yes         1186  4056
I [2020-01-02 03:02:56,923] [tk_nn_classifier] process test_set [random_anno]
I [2020-01-02 03:03:09,499] [tk_nn_classifier] save result to [res/random_anno.tsv]
label   Prec    Reca     F1
 no     0.552   0.568   0.560
 yes    0.942   0.938   0.940
I [2020-01-02 03:03:09,572] [tk_nn_classifier] Confusion matrix:
Predicted   no   yes
Actual
no         100    76
yes         81  1232

* keras models

- models/keras/keras_multi_v1_1024_0.001/best_model.19-0.32.h5
I [2020-01-04 17:49:06,239] [tk_nn_classifier] save result to [keras_res/eval.tsv]
label   Prec    Reca     F1
 yes    0.782   0.846   0.813
 no     0.924   0.888   0.906
I [2020-01-04 17:49:06,810] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         13043  1638
yes         1070  5870

I [2020-01-04 17:49:25,662] [tk_nn_classifier] save result to [keras_res/eval_us.tsv]
label   Prec    Reca     F1
 yes    0.778   0.787   0.782
 no     0.924   0.921   0.923
I [2020-01-04 17:49:25,927] [tk_nn_classifier] Confusion matrix:
Predicted    no  yes
Actual
no         1346  116
yes         110  406

I [2020-01-04 17:49:50,934] [tk_nn_classifier] save result to [keras_res/eval_uk.tsv]
label   Prec    Reca     F1
 yes    0.862   0.964   0.910
 no     0.973   0.894   0.931
I [2020-01-04 17:49:51,049] [tk_nn_classifier] Confusion matrix:
Predicted    no   yes
Actual
no         1528   182
yes          43  1139

I [2020-01-04 17:52:19,512] [tk_nn_classifier] save result to [keras_res/eval_other.tsv]
label   Prec    Reca     F1
 yes    0.763   0.825   0.793
 no     0.917   0.884   0.900
I [2020-01-04 17:52:19,821] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         10169  1340
yes          917  4325

I [2020-01-04 17:52:31,668] [tk_nn_classifier] save result to [keras_res/random_anno.tsv]
label   Prec    Reca     F1
 no     0.604   0.460   0.523
 yes    0.930   0.960   0.945
I [2020-01-04 17:52:31,716] [tk_nn_classifier] Confusion matrix:
Predicted  no   yes
Actual
no         81    95
yes        53  1260

- models/keras/keras_multi_v2_1024_0.001/best_model.14-0.36.h5
I [2020-01-05 03:21:58,039] [tk_nn_classifier] save result to [res/eval.tsv]
label   Prec    Reca     F1
 no     0.928   0.904   0.916
 yes    0.808   0.852   0.830
I [2020-01-05 03:21:58,341] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         13276  1405
yes         1026  5914
I [2020-01-05 03:21:58,347] [tk_nn_classifier] process test_set [eval_us]
I [2020-01-05 03:22:24,902] [tk_nn_classifier] save result to [res/eval_us.tsv]
label   Prec    Reca     F1
 no     0.942   0.936   0.939
 yes    0.823   0.835   0.829
I [2020-01-05 03:22:24,959] [tk_nn_classifier] Confusion matrix:
Predicted    no  yes
Actual
no         1369   93
yes          85  431
I [2020-01-05 03:22:24,964] [tk_nn_classifier] process test_set [eval_uk]
I [2020-01-05 03:22:58,717] [tk_nn_classifier] save result to [res/eval_uk.tsv]
label   Prec    Reca     F1
 no     0.972   0.898   0.933
 yes    0.867   0.963   0.912
I [2020-01-05 03:22:58,845] [tk_nn_classifier] Confusion matrix:
Predicted    no   yes
Actual
no         1535   175
yes          44  1138

I [2020-01-05 03:22:58,850] [tk_nn_classifier] process test_set [eval_other]
I [2020-01-05 03:26:22,465] [tk_nn_classifier] save result to [res/eval_other.tsv]
label   Prec    Reca     F1
 no     0.920   0.901   0.911
 yes    0.793   0.829   0.810
I [2020-01-05 03:26:22,704] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         10372  1137
yes          897  4345
I [2020-01-05 03:26:22,710] [tk_nn_classifier] process test_set [random_anno]
I [2020-01-05 03:26:38,720] [tk_nn_classifier] save result to [res/random_anno.tsv]
label   Prec    Reca     F1
 no     0.591   0.460   0.518
 yes    0.930   0.957   0.943
I [2020-01-05 03:26:38,759] [tk_nn_classifier] Confusion matrix:
Predicted  no   yes
Actual
no         81    95
yes        56  1257


- models/keras/keras_multi_v2_1024_0.003/best_model.67-0.34.h5
I [2020-01-05 03:59:24,090] [tk_nn_classifier] save result to [res_v2_1024_0.003/eval.tsv]
label   Prec    Reca     F1
 no     0.881   0.961   0.920
 yes    0.899   0.726   0.803
I [2020-01-05 03:59:24,648] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         14113   568
yes         1902  5038

I [2020-01-05 03:59:41,670] [tk_nn_classifier] save result to [res_v2_1024_0.003/eval_us.tsv]
label   Prec    Reca     F1
 no     0.889   0.988   0.936
 yes    0.949   0.651   0.772
I [2020-01-05 03:59:41,749] [tk_nn_classifier] Confusion matrix:
Predicted    no  yes
Actual
no         1444   18
yes         180  336

I [2020-01-05 04:00:03,385] [tk_nn_classifier] save result to [res_v2_1024_0.003/eval_uk.tsv]
label   Prec    Reca     F1
 no     0.954   0.972   0.963
 yes    0.958   0.932   0.945
I [2020-01-05 04:00:03,486] [tk_nn_classifier] Confusion matrix:
Predicted    no   yes
Actual
no         1662    48
yes          80  1102

I [2020-01-05 04:02:14,290] [tk_nn_classifier] save result to [res_v2_1024_0.003/eval_other.tsv]
label   Prec    Reca     F1
 no     0.870   0.956   0.911
 yes    0.878   0.687   0.771
I [2020-01-05 04:02:14,587] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         11007   502
yes         1642  3600

I [2020-01-05 04:02:24,841] [tk_nn_classifier] save result to [res_v2_1024_0.003/random_anno.tsv]
label   Prec    Reca     F1
 no     0.414   0.625   0.498
 yes    0.946   0.881   0.912
I [2020-01-05 04:02:24,888] [tk_nn_classifier] Confusion matrix:
Predicted   no   yes
Actual
no         110    66
yes        156  1157

- models/keras/keras_multi_v3_2layer_1024_0.003/best_model.107-0.31.h5
I [2020-01-05 03:24:31,845] [tk_nn_classifier] save result to [res/eval.tsv]
label   Prec    Reca     F1
 no     0.896   0.950   0.922
 yes    0.878   0.767   0.819
I [2020-01-05 03:24:32,175] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         13944   737
yes         1620  5320

I [2020-01-05 03:24:58,463] [tk_nn_classifier] save result to [res/eval_us.tsv]
label   Prec    Reca     F1
 no     0.900   0.977   0.937
 yes    0.913   0.692   0.787
I [2020-01-05 03:24:58,553] [tk_nn_classifier] Confusion matrix:
Predicted    no  yes
Actual
no         1428   34
yes         159  357

I [2020-01-05 03:25:31,999] [tk_nn_classifier] save result to [res/eval_uk.tsv]
label   Prec    Reca     F1
 no     0.959   0.956   0.958
 yes    0.936   0.942   0.939
I [2020-01-05 03:25:32,115] [tk_nn_classifier] Confusion matrix:
Predicted    no   yes
Actual
no         1634    76
yes          69  1113

I [2020-01-05 03:28:52,144] [tk_nn_classifier] save result to [res/eval_other.tsv]
label   Prec    Reca     F1
 no     0.887   0.946   0.915
 yes    0.860   0.734   0.792
I [2020-01-05 03:28:52,375] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         10882   627
yes         1392  3850

I [2020-01-05 03:29:07,752] [tk_nn_classifier] save result to [res/random_anno.tsv]
label   Prec    Reca     F1
 yes    0.939   0.911   0.925
 no     0.456   0.557   0.501
I [2020-01-05 03:29:07,821] [tk_nn_classifier] Confusion matrix:
Predicted   no   yes
Actual
no          98    78
yes        117  1196


* keras v4:
  padding = same,
  layer 2 or less,
  2 dense layers: cnn output => 8, 8 => 1
  evaluated on the best model

- keras_multi_v4_l1_d2_0.001: one layer is less than 2 layer

-   models/keras/keras_multi_v4_l2_d2_0.003/best_model.61-0.28.h5
I [2020-01-05 15:15:16,584] [tk_nn_classifier] save result to [res_l2d2_0.0003/eval.tsv]
label   Prec    Reca     F1
 yes    0.838   0.832   0.835
 no     0.921   0.924   0.922
I [2020-01-05 15:15:16,971] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         13563  1118
yes         1163  5777

I [2020-01-05 15:15:42,230] [tk_nn_classifier] save result to [res_l2d2_0.0003/eval_us.tsv]
label   Prec    Reca     F1
 yes    0.895   0.806   0.848
 no     0.934   0.966   0.950
I [2020-01-05 15:15:42,265] [tk_nn_classifier] Confusion matrix:
Predicted    no  yes
Actual
no         1413   49
yes         100  416

I [2020-01-05 15:16:13,794] [tk_nn_classifier] save result to [res_l2d2_0.0003/eval_uk.tsv]
label   Prec    Reca     F1
 yes    0.911   0.957   0.934
 no     0.969   0.936   0.952
I [2020-01-05 15:16:13,888] [tk_nn_classifier] Confusion matrix:
Predicted    no   yes
Actual
no         1600   110
yes          51  1131

I [2020-01-05 15:16:13,892] [tk_nn_classifier] process test_set [eval_other]
label   Prec    Reca     F1
 yes    0.815   0.807   0.811
 no     0.912   0.917   0.915
I [2020-01-05 15:19:26,899] [tk_nn_classifier] Confusion matrix:
Predicted     no   yes
Actual
no         10550   959
yes         1012  4230

I [2020-01-05 15:19:41,812] [tk_nn_classifier] save result to [res_l2d2_0.0003/random_anno.tsv]
label   Prec    Reca     F1
 yes    0.942   0.943   0.943
 no     0.571   0.568   0.570
I [2020-01-05 15:19:41,887] [tk_nn_classifier] Confusion matrix:
Predicted   no   yes
Actual
no         100    76
yes         75  1238
