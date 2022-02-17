import paddle
import pgl

edges = paddle.to_tensor([[0, 1], [1, 2], [3, 2], [1, 3]])
feat = paddle.randn([4, 128])
g = pgl.Graph(edges, node_feat={'h': feat})
out = pgl.nn.GCNConv(128, 10)(g, g.node_feat['h'])
