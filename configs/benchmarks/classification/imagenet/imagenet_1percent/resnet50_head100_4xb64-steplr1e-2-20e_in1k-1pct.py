_base_ = 'resnet50_head1_4xb64-steplr1e-1-20e_in1k-1pct.py'

# optimizer
optimizer = dict(lr=0.01, paramwise_options={'\\Ahead.': dict(lr_mult=100)})
