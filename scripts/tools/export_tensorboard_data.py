from tensorboard.backend.event_processing import event_accumulator
 
# 加载日志数据
path = "results/vq-experiments/08-10T15-33-54_vqgan_ffhq_f8_vq-cosine/tensorboard/version_0/events.out.tfevents.1660116851.1886f88e80ab.99.0"
ea = event_accumulator.EventAccumulator(path) 
ea.Reload()
print(ea.scalars.Keys())
 
value = ea.scalars.Items('val_rec_loss_epoch')
print(len(value))
# print([(i.step,i.value) for i in val_psnr])
print([i.value for i in value])
print(value[49])