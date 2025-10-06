import torch

class CustomLRScheduler:
    def __init__(self, optimizer, init_lr, warmup_steps):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.cur_lr = init_lr
        self.warmup_steps = warmup_steps
        self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step < self.warmup_steps:
            # 线性预热阶段
            self.cur_lr = self.init_lr * float(self.global_step) / float(self.warmup_steps)
        else:
            # 训练阶段
            self.cur_lr = self.init_lr * ((float(self.warmup_steps) / float( self.global_step)) ** 0.5)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.cur_lr

if __name__ == '__main__':

    # 假设你的模型是model
    model = torch.nn.Linear(10, 2)  # 仅作为示例，这里使用一个简单的线性模型

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0)  # 初始学习率设置为0

    # 初始化你的自定义学习率调度器
    init_lr = 0.001  # 初始学习率
    warmup_steps = 1000  # 预热步数
    scheduler = CustomLRScheduler(optimizer, init_lr, warmup_steps)

    # 训练循环
    for epoch in range(num_epochs):
        for batch in data_loader:
            # ... 这里是你的训练代码 ...

            # 梯度更新
            optimizer.step()

            # 更新学习率
            scheduler.step()

            # 清空梯度
            optimizer.zero_grad()
