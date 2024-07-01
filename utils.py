


def log_print(f_name,str_text):
    with open(f_name,'a') as f:
        f.write(str(str_text)+'\n')

def save_model(model,val_acc,epoch,fold,pathname,f_name):
    log_print(f_name,'saving....')
    s_model=model.to('cpu')
    state = {
        'net': s_model.state_dict(),
        'epoch': epoch,
        'acc': val_acc
    }
    torch.save(state,pathname)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)