# 文本向量部署

这里保存Qwen3-Embedding-0.6B的环境配置、节点配置、集群配置、启动脚本和systemd服务。

## Independent runtime layout

Install this model under /opt/gta-ai/qwen-embedding. Before starting systemd, create config/, data/ and logs/ in that directory. Keep the node runtime configuration in config/node.env and preserve its environment-specific ports and resource limits.

Install deploy/gta-ai-embedding.service into /etc/systemd/system/ and deploy/logrotate.conf into /etc/logrotate.d/gta-ai-embedding. Run systemctl daemon-reload, then enable --now gta-ai-embedding. The console discovers */logs/*.log. Model weights and runtime configuration are not build artifacts.
