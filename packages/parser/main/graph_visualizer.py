import json
from import_graph import ImportGraph
from typing import Any, Deque
from collections import deque


class GraphVisualizer:
    """Import 그래프를 HTML로 시각화"""

    def __init__(self, graph: ImportGraph):
        self.graph = graph

    def generate_html(self, output_path: str):
        """그래프를 인터랙티브 HTML로 생성"""
        html_content = self._create_html()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✓ Graph visualization saved to: {output_path}")

    def _create_html(self) -> str:
        """HTML 콘텐츠 생성"""
        nodes_data = self._prepare_nodes_data()
        edges_data = self._prepare_edges_data()

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Structure Graph Analyzer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
        }}
        .main-container {{
            display: flex;
            height: 100vh;
        }}
        .graph-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        #network {{
            flex: 1;
            width: 100%;
            height: 100%;
        }}
        .sidebar {{
            width: 350px;
            height: 100vh;
            background: white;
            overflow-y: auto;
            padding: 20px;
            box-shadow: -2px 0 8px rgba(0, 0, 0, 0.1);
            border-right: 1px solid #ddd;
        }}
        .detail-panel {{
            width: 400px;
            height: 100vh;
            background: white;
            overflow-y: auto;
            padding: 20px;
            box-shadow: -2px 0 8px rgba(0, 0, 0, 0.1);
            border-left: 2px solid #667eea;
        }}
        .detail-panel.hidden {{
            display: none;
        }}
        .panel-header {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .detail-section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        .section-title::before {{
            content: '';
            display: inline-block;
            width: 4px;
            height: 4px;
            background: #667eea;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .source-code {{
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            overflow-x: auto;
            max-height: 200px;
        }}
        .source-code.empty {{
            color: #999;
            font-style: italic;
        }}
        .import-list {{
            background: #f9f9f9;
            border-left: 3px solid #4facfe;
            padding: 10px;
            border-radius: 4px;
        }}
        .import-item {{
            padding: 5px 0;
            color: #333;
            font-size: 13px;
        }}
        .import-item::before {{
            content: '→ ';
            color: #4facfe;
            font-weight: bold;
        }}
        .no-imports {{
            color: #999;
            font-size: 12px;
            font-style: italic;
        }}
        .sidebar h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .node-info {{
            background: #f8f9fa;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .node-info:hover {{
            background: #e8eaf6;
            transform: translateX(5px);
        }}
        .node-info.active {{
            background: #e8eaf6;
            border-left-color: #764ba2;
        }}
        .node-type {{
            display: inline-block;
            padding: 2px 8px;
            background: #667eea;
            color: white;
            border-radius: 3px;
            font-size: 11px;
            margin-right: 5px;
            font-weight: bold;
        }}
        .node-name {{
            font-weight: 600;
            color: #333;
            margin-top: 5px;
            word-break: break-all;
        }}
        .node-lines {{
            color: #999;
            font-size: 12px;
            margin-top: 3px;
        }}
        .stats {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 13px;
        }}
        .stats-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }}
        .stats-item:last-child {{
            margin-bottom: 0;
        }}
        .stats-label {{
            color: #666;
        }}
        .stats-value {{
            font-weight: bold;
            color: #667eea;
        }}
        .legend {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 13px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        .legend-item:last-child {{
            margin-bottom: 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="graph-container">
            <div id="network"></div>
        </div>
        <div class="sidebar">
            <h3>📊 Statistics</h3>
            <div class="stats">
                <div class="stats-item">
                    <span class="stats-label">Total Nodes:</span>
                    <span class="stats-value">{len(self.graph.nodes)}</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Total Edges:</span>
                    <span class="stats-value">{sum(len(v) for v in self.graph.edges.values())}</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Modules:</span>
                    <span class="stats-value">{self._count_type("module")}</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Classes:</span>
                    <span class="stats-value">{self._count_type("class")}</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Functions:</span>
                    <span class="stats-value">{self._count_type("function")}</span>
                </div>
                <div class="stats-item">
                    <span class="stats-label">Imports:</span>
                    <span class="stats-value">{self._count_type("import")}</span>
                </div>
            </div>

            <h3>🎨 Legend</h3>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #667eea;"></div>
                    <span>Module</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #764ba2;"></div>
                    <span>Class</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f093fb;"></div>
                    <span>Function</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4facfe;"></div>
                    <span>Import</span>
                </div>
            </div>

            <h3>📋 Nodes</h3>
            <div id="nodesList"></div>
        </div>
        <div class="detail-panel hidden" id="detailPanel">
            <div class="panel-header">📄 Node Details</div>
            <div class="detail-section">
                <div class="section-title">Name</div>
                <div id="detailName"></div>
            </div>
            <div class="detail-section">
                <div class="section-title">Type</div>
                <div id="detailType"></div>
            </div>
            <div class="detail-section">
                <div class="section-title">Lines</div>
                <div id="detailLines"></div>
            </div>
            <div class="detail-section" id="importsSection" style="display:none;">
                <div class="section-title">🔗 Imports</div>
                <div id="detailImports"></div>
            </div>
            <div class="detail-section" id="dependenciesSection" style="display:none;">
                <div class="section-title">🔀 Dependencies</div>
                <div id="detailDependencies"></div>
            </div>
            <div class="detail-section" id="sourceSection" style="display:none;">
                <div class="section-title">📝 Source Code</div>
                <pre class="source-code" id="detailSource"></pre>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        const nodesArray = {nodes_data};
        const edgesArray = {edges_data};
        
        const nodesData = new vis.DataSet(nodesArray);
        const edgesData = new vis.DataSet(edgesArray);

        const container = document.getElementById('network');
        const data = {{
            nodes: nodesData,
            edges: edgesData
        }};

        const options = {{
            physics: {{
                enabled: true,
                stabilization: {{
                    enabled: true,
                    iterations: 300,
                    updateInterval: 50
                }},
                barnesHut: {{
                    gravitationalConstant: -20000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.05,
                }},
                minVelocity: 0.75
            }},
            interaction: {{
                hover: true,
                navigationButtons: true,
                keyboard: true,
                zoomView: true,
                dragNodes: true
            }},
            nodes: {{
                font: {{
                    size: 13,
                    color: '#ffffff',
                    face: 'Segoe UI'
                }},
                borderWidth: 2,
                borderWidthSelected: 3,
                shadow: {{
                    enabled: true,
                    color: 'rgba(0,0,0,0.2)',
                    size: 10,
                    x: 5,
                    y: 5
                }}
            }},
            edges: {{
                smooth: {{
                    type: 'continuous'
                }},
                color: {{
                    color: '#cccccc',
                    highlight: '#667eea'
                }},
                font: {{ size: 11 }},
                width: 1.5,
                arrows: {{
                    to: {{ enabled: true, scaleFactor: 0.8 }}
                }}
            }}
        }};

        const network = new vis.Network(container, data, options);

        // After stabilization, stop physics so layout stays put
        network.once('stabilizationIterationsDone', function() {{
            network.setOptions({{ physics: false }});
            console.log('Physics stabilized and disabled');
        }});

        // 노드 클릭 이벤트
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodesData.get(nodeId);
                highlightNode(nodeId);
                showNodeDetails(node);
            }}
        }});

        function highlightNode(nodeId) {{
            const allNodes = document.querySelectorAll('.node-info');
            allNodes.forEach(n => n.classList.remove('active'));
            const element = document.querySelector(`[data-node-id="${{nodeId}}"]`);
            if (element) {{
                element.classList.add('active');
                element.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }}
        }}

        function showNodeDetails(node) {{
            const detailPanel = document.getElementById('detailPanel');
            document.getElementById('detailName').textContent = node.full_name;
            document.getElementById('detailType').textContent = node.type.toUpperCase();
            document.getElementById('detailLines').textContent = node.lines;
            
            // Import 정보 표시
            const importsSection = document.getElementById('importsSection');
            const detailImports = document.getElementById('detailImports');
            if (node.imports && node.imports.length > 0) {{
                importsSection.style.display = 'block';
                detailImports.innerHTML = '';
                node.imports.forEach(imp => {{
                    const div = document.createElement('div');
                    div.className = 'import-item';
                    div.textContent = imp;
                    detailImports.appendChild(div);
                }});
            }} else {{
                importsSection.style.display = 'none';
            }}
            
            // 클래스 의존성 표시
            const dependenciesSection = document.getElementById('dependenciesSection');
            const detailDependencies = document.getElementById('detailDependencies');
            if (node.dependencies && node.dependencies.length > 0) {{
                dependenciesSection.style.display = 'block';
                detailDependencies.innerHTML = '';
                node.dependencies.forEach(dep => {{
                    const div = document.createElement('div');
                    div.className = 'import-item';
                    div.style.color = '#764ba2';
                    div.innerHTML = `<strong>◆</strong> ${{dep}}`;
                    detailDependencies.appendChild(div);
                }});
            }} else {{
                dependenciesSection.style.display = 'none';
            }}
            
            // 소스코드 표시
            const sourceSection = document.getElementById('sourceSection');
            const detailSource = document.getElementById('detailSource');
            if (node.source_code && node.source_code.trim()) {{
                sourceSection.style.display = 'block';
                detailSource.textContent = node.source_code;
                detailSource.classList.remove('empty');
            }} else {{
                sourceSection.style.display = 'block';
                detailSource.textContent = '(No source code)';
                detailSource.classList.add('empty');
            }}
            
            detailPanel.classList.remove('hidden');
        }}

        // 노드 목록 생성
        const nodesList = document.getElementById('nodesList');
        const sortedNodes = Array.from(nodesData.get()).sort((a, b) => 
            a.label.localeCompare(b.label)
        );

        sortedNodes.forEach(node => {{
            const div = document.createElement('div');
            div.className = 'node-info';
            div.setAttribute('data-node-id', node.id);
            div.innerHTML = `
                <span class="node-type">${{node.type}}</span>
                <div class="node-name">${{node.label}}</div>
                <div class="node-lines">${{node.lines}}</div>
            `;
            div.addEventListener('click', () => {{
                network.selectNodes([node.id]);
                network.focus(node.id, {{ scale: 2, animation: true }});
                highlightNode(node.id);
                showNodeDetails(node);
            }});
            nodesList.appendChild(div);
        }});
    </script>
</body>
</html>
"""
        return html

    def _prepare_nodes_data(self) -> str:
        """노드 데이터를 JSON 형식으로 준비"""
        nodes: list[dict[str, Any]] = []
        type_colors = {
            "module": "#667eea",
            "class": "#764ba2",
            "function": "#f093fb",
            "import": "#4facfe",
        }

        # --- Build adjacency and compute levels with BFS ---
        # Use only edges that point to existing nodes
        adjacency: dict[str, list[str]] = {}
        for src, targets in self.graph.edges.items():
            adjacency[src] = [t for t in targets if t in self.graph.nodes]

        # start BFS from module nodes (level 0)
        levels: dict[str, int] = {}
        q: Deque[str] = deque()
        # enqueue all module nodes
        for node_id, block in self.graph.nodes.items():
            if block.block_type == "module":
                levels[node_id] = 0
                q.append(node_id)

        # BFS
        while q:
            cur = q.popleft()
            cur_level = levels[cur]
            for nei in adjacency.get(cur, []):
                if nei not in levels:
                    levels[nei] = cur_level + 1
                    q.append(nei)

        # Assign remaining nodes (disconnected) to increasing levels
        max_level = max(levels.values()) if levels else 0
        for node_id in self.graph.nodes:
            if node_id not in levels:
                max_level += 1
                levels[node_id] = max_level

        # Group nodes by level
        nodes_by_level: dict[int, list[str]] = {}
        for nid, lvl in levels.items():
            nodes_by_level.setdefault(lvl, []).append(nid)

        # Append nodes (positions will be handled by vis.js physics)
        for lvl, nids in nodes_by_level.items():
            # sort nodes for stable layout (by type then name)
            nids.sort(key=lambda x: (self.graph.nodes[x].block_type, x))
            for nid in nids:
                block = self.graph.nodes[nid]
                color = type_colors.get(block.block_type, "#999999")
                node_obj: dict[str, Any] = {
                    "id": nid,
                    "label": nid.split(".")[-1],
                    "title": nid,
                    "color": color,
                    "type": block.block_type,
                    "lines": f"Lines {block.start_line}-{block.end_line}",
                    "source_code": block.source_code or "",
                    "imports": block.imports or [],
                    "dependencies": block.dependencies or [],
                    "full_name": nid,
                }
                nodes.append(node_obj)

        return json.dumps(nodes)

    def _prepare_edges_data(self) -> str:
        """엣지 데이터를 JSON 형식으로 준비"""
        edges: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for from_node, to_nodes in self.graph.edges.items():
            for to_node in to_nodes:
                edge_key = (from_node, to_node)
                if edge_key not in seen:
                    edges.append(
                        {
                            "from": from_node,
                            "to": to_node,
                            "label": "",
                        }
                    )
                    seen.add(edge_key)

        return json.dumps(edges)

    def _count_type(self, block_type: str) -> int:
        """특정 타입의 블록 개수 반환"""
        return sum(
            1 for block in self.graph.nodes.values() if block.block_type == block_type
        )
