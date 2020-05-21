

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform

from keras.models import Sequential
from keras.layers import Dropout, TimeDistributed, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from ClockWorkrnn import ClockworkRNN

main_debug = False
cwrnn_debug = False


def run(col, epochs, look_back, period_spec=None, units=32):


    if period_spec is None:
        period_spec = [1, 2, 4, 8]
        # period_spec = [8]
        # period_spec = [4, 8, 15, 30, 60]

    os = platform.architecture()[1]
    if os == 'WindowsPE':
        if epochs is None:
            epochs = 1
    else:
        if epochs is None:
            epochs = 10
        matplotlib.use('Agg')
        plt.switch_backend('agg')
        if main_debug:
            print("For Linux ---------------- !")

    plt.rcParams.update({'figure.autolayout': True})

    def create_dataset(dataset_in, look_back_in):
        # type: (np.array, int) -> object
        """
        look_back = 2  <----after change, size will different
        """
        data_x, data_y = [], []
        for ii in range(len(dataset_in) - look_back_in - 1):
            a = dataset_in[ii:(ii + look_back_in), :]
            data_x.append(a)
            data_y.append(dataset_in[ii + look_back_in, :])
        return np.array(data_x), np.array(data_y)

    #  read data
    def read_data(file_location, sheetname):

        data = pd.read_excel(file_location, sheet_name=sheetname)
        altitude = data.altitude._values
        speed = data.speed._values
        flightTime = data.flightTime._values
        xz = data.x._values
        yz = data.y._values
        vy = data.vy._values
        or_data = np.vstack((altitude, xz, yz, speed, vy)).T

        start = []
        for i in range(1, len(flightTime)):
            if flightTime[i] - flightTime[i - 1] < 0 or flightTime[i] - flightTime[i - 1] > 100:
                start.append(i)

        sq_set = [[] for i in range(len(start) + 1)]

        for i in range(0, len(start) - 1):
            for j in range(start[i], start[i + 1]):
                sq_set[i + 1].append(or_data[j])

        for j in range(0, start[0]):
            sq_set[0].append(or_data[j])

        for j in range(start[-1], len(flightTime)):
            sq_set[-1].append(or_data[j])

        # 以前一个点作为原点，计算坐标
        sq_set_diff = [[] for i in range(len(sq_set))]
        for i in range(0, len(sq_set)):
            sq_set_diff[i] = [[[] for l in range(5)]for k in range(len(sq_set[i]))]
            sq_set_diff[i][0][0] = 0
            sq_set_diff[i][0][1] = 0
            sq_set_diff[i][0][2] = 0
            sq_set_diff[i][0][3] = 0
            sq_set_diff[i][0][4] = 0
            for j in range(1, len(sq_set[i])):
                sq_set_diff[i][j][0] = sq_set[i][j][0] - sq_set[i][j-1][0]
                sq_set_diff[i][j][1] = sq_set[i][j][1] - sq_set[i][j-1][1]
                sq_set_diff[i][j][2] = sq_set[i][j][2] - sq_set[i][j-1][2]
                sq_set_diff[i][j][3] = sq_set[i][j][3]
                sq_set_diff[i][j][4] = sq_set[i][j][4]

        # 将各条轨迹合并
        data_combine = np.array(sq_set_diff[0])
        for i in range(1, len(sq_set_diff)):
            data_i = np.array(sq_set_diff[i])
            combine = np.vstack((data_combine, data_i))
            data_combine = combine
        return data_combine

    dataset = read_data(r"D:\轨迹预测\2.xlsx", '四旋翼')
    # df = pd.DataFrame(dataset)
    # df.to_csv(r"D:\轨迹预测\diff_xyz.csv")

    dataset = dataset[0:40000, :]
    dataset = dataset.astype('float32')

    np.random.seed(7)  # fix random seed for reproducibility

    scaler = MinMaxScaler(feature_range=(0, 1))  # normalize the dataset
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    # test_size = len(dataset) - train_size

    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # use this function to prepare the train and test datasets for modeling

    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 5, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 5, test_x.shape[1]))

    # if main_debug:
    #     print(train_y)

    # reshape_train_y = []
    # for i in train_y.tolist():
    #     reshape_train_y.append([[i]])
    # if main_debug:
    #     print("------------------------------------------")
    #     print(reshape_train_y)
    # train_y = np.array(reshape_train_y)
    #
    # reshape_test_y = []
    # for i in test_y.tolist():
    #     reshape_test_y.append([[i]])
    # test_y = np.array(reshape_test_y)
    #
    # if main_debug:
    #     print(train_x.shape)
    #     print(test_x.shape)
    #     print(train_y.shape)
    #     print(test_y.shape)
    train_y = np.reshape(train_y, (train_x.shape[0], train_x.shape[1], 1))
    test_y = np.reshape(test_y, (test_x.shape[0], test_x.shape[1], 1))

    # =================================== Model Define Core ===============================================

    model = Sequential()

    model.add(ClockworkRNN(units=units,
                           period_spec=period_spec,
                           input_shape=train_x.shape[1:],  # (timesteps, dimension)<-----(samples, timesteps, dimension)
                           dropout_W=0.2,
                           return_sequences=True))  # debug is for developing mode, you can remove
    model.add(Dropout(.2))

    model.add(TimeDistributed(Dense(units=1, activation='linear')))
    # Dense(units:cell_number)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  # adam, sgd

    model.fit(train_x, train_y, epochs=epochs, batch_size=1, verbose=1)

    train_predict = model.predict(train_x)  # 得到的shape是还是三维的[samples, input_size, units=1]
    test_predict = model.predict(test_x)

    tmp_train_p = []
    for i in train_predict.tolist():  # array.tolist 转换为list, 长度就是第一个维度，第二维度是其中一个list的长度，以此类推，不存在array了
        tmp_train_p.append(i)  # 所以i就是一个list，长度是第二维度,i[0]也是一个list，长度是第三维度
    train_predict = np.array(tmp_train_p)

    tmp_test_p = []
    for i in test_predict.tolist():
        tmp_test_p.append(i)
    test_predict = np.array(tmp_test_p)

    if main_debug:
        print(train_predict)
        print(test_predict)
        print(type(test_predict))
        print(test_predict.shape)

    def print_array(nd_array_in):
        print(nd_array_in)
        print(type(nd_array_in))
        print(nd_array_in.shape)

    if main_debug:
        print_array(train_y)

    # tmp_train__y = []
    # for i in train_y:
    #     tmp_train__y.append(i[0][0])
    # train_y = np.array(tmp_train__y)
    #
    # tmp_test__y = []
    # for i in test_y:
    #     tmp_test__y.append(i[0][0])
    # test_y = np.array(tmp_test__y)

    # invert predictions
    train_predict = scaler.inverse_transform(train_predict.squeeze())
    train_y = scaler.inverse_transform(train_y.squeeze())
    test_predict = scaler.inverse_transform(test_predict.squeeze())
    test_y = scaler.inverse_transform(test_y.squeeze())

    train_score = math.sqrt(mean_squared_error(train_y[:, col], train_predict[:, col]))

    test_score = math.sqrt(mean_squared_error(test_y[:, col], test_predict[:, col]))

    print('Test Score: %.2f RMSE' % test_score)
    print('Train Score: %.2f RMSE' % train_score)

    # shift train predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    # shift test predictions for plotting
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict

    if main_debug:
        print('=============********* test_predict **********==============')
        print(test_predict_plot)
        print('=============********* actual list ***********==============')
        print(scaler.inverse_transform(dataset))

    train_list = list(train_predict_plot.tolist())
    predict_list = list(test_predict_plot.tolist())
    actual_list = list(scaler.inverse_transform(dataset).tolist())

    if main_debug:
        print('==================== find compare ==========================')

    def find_mape(predict_list_in, actual_list_in):
        # type: (list, list) -> float
        mape_and_weight_error = []
        total_order_in_predict_period = 0.0
        for iiii in range(len(predict_list_in)):
            forecast = predict_list_in[iiii][0]
            actual = actual_list_in[iiii][0]
            if math.isnan(forecast):
                pass
            else:
                total_order_in_predict_period += actual
                mape = abs((actual - forecast) / (actual + 0.001))
                mae = abs(actual - forecast)
                weight_error = mape
                mape_and_weight_error.append([actual, forecast, weight_error, mape, mae])
        mean_mape = np.mean(np.array(mape_and_weight_error[3]))
        mean_mae = np.mean(np.array(mape_and_weight_error[4]))
        weight_error_total = 0.0
        for iiii in mape_and_weight_error:
            weight_error_total += (iiii[2] * iiii[0] / total_order_in_predict_period)
        result_mape_total = 1 - weight_error_total
        return mean_mape, result_mape_total, mean_mae


    mape_mean_predict, mape_total_predict, mae_mean_predict = find_mape(predict_list, actual_list)
    mape_mean_train, mape_total_train, mae_mean_train = find_mape(train_list, actual_list)
    # plot baseline and predictions
    f, (ax1) = plt.subplots(figsize=(10, 4), nrows=1)
    data_org = scaler.inverse_transform(dataset)
    ax1.plot(data_org[:, col], label="original data")
    ax1.plot(train_predict_plot[:, col], label="training_" + str(mape_total_train * 100.0) + " %")
    ax1.plot(test_predict_plot[:, col], label="predict_" + str(mape_total_predict * 100.0) + " %")
    ax1.set_title('period_spec='+str(period_spec)+'; units='+str(units), fontsize=12, color='black')

    handles, labels = ax1.get_legend_handles_labels()
    # reverse the order
    ax1.legend(handles[::-1], labels[::-1])

    if os == 'WindowsPE':
        plt.show()
    else:
        plt.savefig("result_"+str(epochs)+"_"+str(len(period_spec)) + "_" + str(units) + ".jpg")

    f2, (ax2) = plt.subplots(figsize=(10, 4), nrows=1)
    data_org = scaler.inverse_transform(dataset)
    ax2.plot(data_org[39000:, col], label="original data")
    ax2.plot(test_predict_plot[39000:, col], label="predict_" + str(mape_total_predict * 100.0) + " %")
    ax2.set_title('period_spec=' + str(period_spec) + '; units=' + str(units), fontsize=12, color='black')
    plt.legend()
    plt.show()

    return mape_mean_train, mape_total_train, mae_mean_train, mape_mean_predict, mape_total_predict,  mae_mean_predict


if __name__ == "__main__":

    if platform.architecture()[1] == 'WindowsPE':
        train_s = [0.3663521537061736, 0.6779432534824161, 0.8518191177129474, 0.8749907178148822, 0.8641000517580399,
                   0.8859109384626707, 0.9104095038141398, 0.8813243499784189, 0.9248637281561507, 0.9034520473822252,
                   0.8996140640360283, 0.8814416195461714, 0.9155803367698807, 0.9070072317314696, 0.9171993315042655,
                   0.9170290722862964, 0.9023263495457842, 0.8994262257360113, 0.9152705244551164, 0.9008479773310732,
                   0.8501485261122657, 0.9125250148654849, 0.8422393502206864, 0.9246665843565834, 0.9156630290070059,
                   0.9199581465391145, 0.8974542373169994, 0.8815243573984904, 0.9080132586136185, 0.8975249398538326,
                   0.9024474203796928, 0.9216520424355108, 0.8995755830584764, 0.854689041223363, 0.9163160591635147,
                   0.9133246958429556, 0.9180634677643507, 0.9116043400930995, 0.8874409622819113, 0.9194673757092402,
                   0.9240552282411978, 0.9212474731716772, 0.9228283376586673, 0.8788185674923422, 0.9172761241763163,
                   0.9125277832009174, 0.9099940659701126, 0.8917727747033136, 0.9062911542813306, 0.9157131540230472,
                   0.9202419835351451, 0.9056835659809588, 0.9184933737342037, 0.9228623189507613, 0.8960639571646974,
                   0.9069153982094922, 0.9119846076089793, 0.922986141901367, 0.8629763670721553, 0.921851047334988,
                   0.917732450466461, 0.930814983354509, 0.9163368617192631, 0.9230012232981326, 0.9264823549279716,
                   0.9203733096783075, 0.9197947620376976, 0.9139179325896638, 0.9155673344122848, 0.9196398305385411,
                   0.8996416916384511, 0.9011543573559163, 0.9150444950874463, 0.9042486356495513, 0.9072431930473523,
                   0.9104931093688134, 0.9158511197110957, 0.9096137862901812, 0.9106424301991235, 0.8996578830049344,
                   0.9082364385521778, 0.9212050834788458, 0.915421119469842, 0.9176859989940329, 0.9204905204531434,
                   0.9175325180580416, 0.928106387607102, 0.9253782269962071, 0.9175424753480287, 0.9144849963517981,
                   0.9164739749100215, 0.9109830986583659, 0.9146780621530332, 0.9038626925898826, 0.906147646854062,
                   0.9152479702759233, 0.9224023140486485, 0.9102760337568654, 0.9076738460927141, 0.8968587796171852]
        predict_s = [0.10893128967325305, 0.7232950654397687, 0.8230963006533105, 0.8448780429749947,
                     0.8427427416613931, 0.8542806981004901, 0.8763031525753069, 0.7787493148192439, 0.8982002867746096,
                     0.932362475782651, 0.9079817132551435, 0.8588627933403291, 0.8967786491382457, 0.8749128793383815,
                     0.900617057481945, 0.8888680047193076, 0.8660659123861227, 0.900394229261767, 0.8938537998780121,
                     0.8915564262802724, 0.8313104043338899, 0.8808749669568362, 0.8254316530050112, 0.9121787935554673,
                     0.9064053454970321, 0.9086273932691504, 0.8347512281757771, 0.8644681223586925, 0.8919573660221295,
                     0.8914820205136538, 0.8970237281385856, 0.9169614845449542, 0.8804824496456514, 0.8420836086453505,
                     0.9036173241159595, 0.9035727859363718, 0.9029257738966444, 0.8838018577410263, 0.8868817621189737,
                     0.8962158751374653, 0.9125643532065315, 0.9212817932453479, 0.9144362109590013, 0.8847244251633849,
                     0.9208564421867651, 0.9163320634840711, 0.9111427247249836, 0.906321871885217, 0.9192757487781181,
                     0.8722645467164536, 0.886823281154697, 0.9114832838782547, 0.9130571981456921, 0.906720692287182,
                     0.8530571111749405, 0.8643840903031045, 0.879945789757363, 0.9069658124841936, 0.8048008129476562,
                     0.9028581426938275, 0.9229787299294965, 0.9356853855951259, 0.8860972127018494, 0.9242774778939804,
                     0.9248341319002995, 0.9052741443966232, 0.8918131135872345, 0.8711676349791785, 0.9017910298824636,
                     0.8907438768585707, 0.8489725160981716, 0.899984102662106, 0.9116952041352304, 0.8587150666527819,
                     0.8969681675605963, 0.9045798751969716, 0.8721597011868074, 0.8438880961072579, 0.8476817557106024,
                     0.8326539147709522, 0.9127652831069615, 0.8920405604120583, 0.9192077284956234, 0.9094290530372818,
                     0.9176227460557047, 0.8880807729411104, 0.9196223318002831, 0.9161540618201833, 0.9111101290028002,
                     0.8805276835912512, 0.903419140660314, 0.8731618651564272, 0.8945393399606572, 0.8987135057566626,
                     0.8570563055511371, 0.8806464627507704, 0.9048126409698612, 0.8538863406712998, 0.9078428301198023,
                     0.8277191531337507]

        t = 0.0
        h = 0
        for jjj in range(len(train_s)):
            if t < (train_s[jjj] + predict_s[jjj]):
                t = (train_s[jjj] + predict_s[jjj])
                h = jjj
        print(t)
        print(h)

        # f_main, (ax_main) = plt.subplots(figsize=(10, 4), nrows=1)
        # ax_main.plot(train_s, label='train')
        # ax_main.plot(predict_s, label='predict')
        # handles_main, labels_main = ax_main.get_legend_handles_labels()
        # ax_main.legend(handles_main[::-1], labels_main[::-1])
        # plt.show()

        units_main = 64
        period_spec_main = [1, 2]  # , 4, 8, 10, 12, 14, 6]
        for iii in range(1, 11):
            print(units_main % iii == 0, units_main, iii)

        epoch = 2
        col_index = 1
        time_step = 50
        mape_mean_train_o, \
        mape_total_train_o, \
        mae_mean_train_o, \
        mape_mean_predict_o, \
        mape_total_predict_o,\
        mae_mean_predict_o = run(col_index, epoch, time_step, period_spec_main, units_main)

        print('------mape_mean_train_o-------')
        print(mape_mean_train_o)
        print('------mape_total_train_o-------')
        print(mape_total_train_o)
        print('------mae_mean_train_o-------')
        print(mae_mean_train_o)
        print('------mape_mean_predict_o-------')
        print(mape_mean_predict_o)
        print('------mape_total_predict_o-------')
        print(mape_total_predict_o)
        print('------mae_mean_predict_o-------')
        print(mae_mean_predict_o)
