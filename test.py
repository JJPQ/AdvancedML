net.load_state_dict(torch.load('W_67_0.8470423817634583_0.7898963689804077.pt'))
# 要指定为eval()，否则会启用drop_out层，带来随机性，使得输出不一致
net.eval()
# 结果为732：268
with open('502023330061.txt', 'a') as f:
    for x in test_data:
        x = x.to(device)
        y_hat = net(x)
        if y_hat >= 0.5:
            f.write('1\n')
        else:
            f.write('0\n')
test_data = torch.tensor(train_data_original.values, dtype=torch.float32)
test_label = torch.tensor(train_labels_original, dtype=torch.float32).reshape(-1,1)
test_dataset = TensorDataset(test_data,test_label)
test_iter = DataLoader(test_dataset,batch_size=128,shuffle=False)
net.eval()
with torch.no_grad():
    total_test_acc, n_test = 0, 0
    tp_t = 0
    fn_t = 0
    fp_t = 0
    tn_t = 0
    for x, y in test_iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        n_test += y.shape[0]
        y_pre = torch.where(y_hat > 0.5, 1, 0)
        test_acc = (y_pre == y).sum()
        total_test_acc += test_acc
        tp_t += y_pre[y.bool()].sum()
        fn_t += (y_pre[y.bool()].shape[0] - y_pre[y.bool()].sum())
        fp_t += y_pre[~y.bool()].sum()
        tn_t += (y_pre[~y.bool()].shape[0] - y_pre[~y.bool()].sum())
    print(
        f'valid epoch:{epoch},tp:{tp_t}, fp:{fp_t}, fn:{fn_t}, tn:{tn_t}, num:{tp_t+fn_t+fp_t+tn_t}'
    )
    f1_positive = 2 * tp_t / (2 * tp_t + fn_t + fp_t)
    f1_negative = 2 * tn_t / (2 * tn_t + fn_t + fp_t)
    macro_f1 = (f1_positive + f1_negative)/2
    print(
        f'f1_positive:{f1_positive},f1_negative:{f1_negative},macro_f1:{macro_f1},acc:{total_test_acc/n_test}, acc_2:{(tp_t+tn_t)/(tp_t+fn_t+fp_t+tn_t)}\n'
    )
# valid epoch:199,tp:1475, fp:1023, fn:383, tn:6119, num:9000
# f1_positive:0.6772268414497375,f1_negative:0.8969510197639465,macro_f1:0.787088930606842,acc:0.8437777757644653