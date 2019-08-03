sample aggregate

pretrain -> 根据不同的任务进行pretrain -> pretrain 可以为图上每一个节点生成一个embedding

但是要求这些embedding能够满足pretrain 任务的某些需求

cluster detection : pretrain 任务 -> 检测子图 ： 为每个节点生成embedding要求能够保留节点所属的子图信息

edge reconstruction : pretrain 任务 -> 能够恢复被mask掉的边： 为每个节点生成的embedding要求能够帮助恢复节点之间的边



betweeness -> 求他的话 一阶邻居是不够的



图上的机器学习任务有些是标签很少的

通过设立另外一种容易拿到标签的任务辅助学习