"""Read-only pipeline visualization. Draws observed data and actual planner events."""
from collections import deque
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = '#0b1220'
CARD = '#142137'
FG = '#e7edf6'
MUTED = '#93a6bf'
CYAN = '#50dbd1'
GOLD = '#ffd071'
RED = '#ff8790'
GREEN = '#8adea0'


def short(key):
    if key is None:
        return '-'
    if key.startswith('instance_'):
        suffix = key.rsplit('_', 1)[1]
        return 'i'+f'{int(suffix):03d}' if suffix.isdecimal() else '…'+key[-4:]
    if '_obj_' in key:
        return key.replace('shelf_', 's').replace('_obj_', '.')
    return key if len(key) <= 5 else '…'+key[-4:]


def region_name(key):
    return {'workspace': 'WORKSPACE', 'buffer_tray': 'BUF', 'output_tray': 'OUT',
            'tabletop': 'TBL', 'unassigned': 'HAND', 'bin': 'BIN'}.get(key, key.replace('shelf_', 'S'))


def xyz(pose):
    return ' / '.join(f'{v:+.3f}' for v in pose.position) + ' m' if pose else '-'


class PipelineInspector:
    width, height = 600, 960

    def __init__(self, *, controller='expert', paused=False):
        self.controller, self.paused = controller, paused
        self.step_until_plan = False
        self.graph = self.task = self.goal = self.observation = None
        self.explanation = {}
        self.status = '관측 대기'
        self.error = None
        self.finished = False
        self.index, self.steps, self.budget, self.snapshots = 0, 0, 0, 0
        self.chunk_shape = None
        self.command = None
        self.plan_time = None
        self.history = []
        self.changes = deque(maxlen=4)
        self.nodes, self.previous_nodes, self.last_regions = {}, {}, {}
        self.added, self.moved = set(), set()
        self._scene_key = None
        self._fonts = {}
        self._cache, self._cache_time = None, 0.
        self.hits = {}
        self.selected = None
        self.camera_image = None

    def font(self, size):
        if size not in self._fonts:
            candidates = ('C:/Windows/Fonts/malgun.ttf', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                          '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
            path = next((p for p in candidates if Path(p).is_file()), None)
            self._fonts[size] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
        return self._fonts[size]

    def invalidate(self):
        self._cache = None

    def handle_key(self, key):
        if self.finished or self.error:
            return
        if key == 'space':
            self.paused = not self.paused
            self.step_until_plan = False
        elif key == 'next':
            self.paused = False
            self.step_until_plan = True
        self.invalidate()

    def select_at(self, x, y):
        self.selected = next((key for key, box in self.hits.items()
                              if box[0] <= x <= box[2] and box[1] <= y <= box[3]), self.selected)
        self.invalidate()

    def observe_scene(self, *, observation, graph, raw_graph=None, camera_views=None):
        self.observation, self.graph = observation, graph
        key = (graph.episode_id, graph.timestamp)
        if key == self._scene_key:
            return
        if self._scene_key and (self._scene_key[0] != graph.episode_id or graph.timestamp < self._scene_key[1]):
            self.nodes, self.last_regions = {}, {}
            self.history.clear()
            self.changes.clear()
            self.snapshots, self.goal, self.finished, self.error = 0, None, False, None
        self._scene_key = key
        self.snapshots += 1
        self.previous_nodes = self.nodes
        nodes = {}
        for obj in graph.nodes:
            parent = obj.parent_id
            while graph.groups.get(parent) not in (None, 'workspace'):
                parent = graph.groups[parent]
            parent = parent or 'unassigned'
            self.last_regions[obj.object_id] = parent
            nodes[obj.object_id] = dict(parent=parent, pos=tuple(obj.pose.position),
                color=obj.color_rgb, status='held' if graph.held_object_id == obj.object_id else 'seen')
        if raw_graph is not None:
            for ident, node in raw_graph.nodes(data=True):
                if node.get('kind') == 'object_candidate' and node.get('status') == 'history' and ident not in nodes:
                    nodes[ident] = dict(parent=self.last_regions.get(ident, 'unassigned'),
                        pos=tuple(node['obb']['center_m']), color=node.get('mean_rgb'), status='history')
        self.nodes = nodes
        if camera_views:
            strip = Image.new('RGB', (420, 180), BG)
            sd = ImageDraw.Draw(strip)
            for i, (name, view) in enumerate(list(camera_views.items())[:2]):
                rgb = view['rgb'].copy()
                for detection in view['detections']:
                    key = detection['object_id']
                    if key in nodes and nodes[key]['color'] is not None:
                        color = np.array(nodes[key]['color'])*255
                        mask = detection['mask']
                        rgb[mask] = (.6*rgb[mask]+.4*color).astype(np.uint8)
                        inner = np.roll(mask, 2, 0) & np.roll(mask, -2, 0) & np.roll(mask, 2, 1) & np.roll(mask, -2, 1)
                        rgb[mask & ~inner] = [80, 235, 220]
                strip.paste(Image.fromarray(rgb).resize((206, 154)), (i*210, 24))
                self._text(sd, (i*210+4, 4), 'A  '+name+' / RGB-D masks', 10, CYAN)
            self.camera_image = strip
        self.added = set(nodes)-set(self.previous_nodes)
        self.moved = {k for k in set(nodes) & self.previous_nodes.keys()
                      if np.linalg.norm(np.array(nodes[k]['pos'])-self.previous_nodes[k]['pos']) > .005}
        absent = sum(v['status'] == 'history' for v in nodes.values())
        if self.snapshots == 1:
            self.changes.appendleft(f't={graph.timestamp:.2f}  최초 생성 · 현재 객체 {len(graph.nodes)}개')
        else:
            switches = [f'{short(k)} {region_name(self.previous_nodes[k]["parent"])}→{region_name(v["parent"])}'
                        for k, v in nodes.items() if k in self.previous_nodes and v['status'] != 'history' and
                        v['parent'] != self.previous_nodes[k]['parent']]
            states = [f'{short(k)} {self.previous_nodes[k]["status"]}→{v["status"]}' for k, v in nodes.items()
                      if k in self.previous_nodes and self.previous_nodes[k]['status'] != v['status']]
            detail = ', '.join((switches or states)[:2]) or f'이동 {len(self.moved)} · 미관측 {absent}'
            self.changes.appendleft(f't={graph.timestamp:.2f}  +{len(self.added)} 객체 · {detail}')
        self.invalidate()

    def on_event(self, event, **data):
        if event == 'scene':
            self.task = data['task']
            self.observe_scene(observation=data['observation'], graph=data['graph'], raw_graph=data.get('raw_graph'))
        elif event == 'planned':
            self.goal, self.task = data['subgoal'], data['task']
            self.observation = data['observation']
            self.plan_time = data['graph'].timestamp
            self.index, self.budget, self.steps = data['index'], data['budget'], 0
            self.explanation = data.get('explanation') or dict(reason='외부 플래너: 결정 이유를 제공하지 않습니다.', support_pairs=[])
            self.status = '실행 중'
            if self.step_until_plan:
                self.paused, self.step_until_plan = True, False
        elif event == 'chunk':
            self.chunk_shape = data['chunk'].commands.shape
        elif event == 'completed':
            self.history.append((data['subgoal'].primitive, data['subgoal'].object_id, data['steps']))
            self.status = '재관측 · 재계획'
        elif event == 'finished':
            self.finished, self.status = True, '작업 완료'
            self.observation = data['observation']
        elif event == 'failed':
            self.error, self.status = data['error'], '실행 중단'
        self.invalidate()

    def on_step(self, before, command, observation, goal):
        self.observation, self.command = observation, command
        self.steps += 1

    def _text(self, draw, pos, text, size=14, color=FG):
        draw.text(pos, str(text), font=self.font(size), fill=color)

    def _wrap(self, draw, pos, text, *, size=14, width=540, lines=2, color=FG):
        x, y = pos
        current, rows = '', []
        for char in str(text):
            if char == '\n':
                rows.append(current)
                current = ''
                continue
            if draw.textlength(current+char, font=self.font(size)) > width:
                rows.append(current)
                current = char
            else:
                current += char
        rows.append(current)
        for i, row in enumerate(rows[:lines]):
            self._text(draw, (x, y+i*(size+5)), row, size, color)

    def _card(self, draw, y1, y2):
        draw.rounded_rectangle((16, y1, 584, y2), radius=12, fill=CARD)

    def _graph(self, draw):
        self._card(draw, 82, 376)
        seen = sum(n['status'] != 'history' for n in self.nodes.values())
        ghosts = len(self.nodes)-seen
        graph_time = self.graph.timestamp if self.graph else 0.
        self._text(draw, (30, 94), f'A  Scene Graph  ·  #{self.snapshots}  ·  t={graph_time:.2f}s', 17)
        self._text(draw, (30, 120), f'현재 {seen}  /  history {ghosts}  ·  영역 계층 (프레임·stack 그룹 접힘)', 12, MUTED)
        groups = {node['parent'] for node in self.nodes.values()}
        if self.graph:
            groups |= {g for g, p in self.graph.groups.items() if p == 'workspace' and not g.startswith('stack:')}
        groups.discard('unassigned') if not any(n['parent'] == 'unassigned' for n in self.nodes.values()) else None
        groups = sorted(groups, key=lambda g: (not g.startswith('shelf_'), g))
        if not groups:
            self._text(draw, (30, 180), '관측한 그래프를 기다리는 중입니다.', 15, MUTED)
            return
        draw.line((300, 150, 300, 157), fill='#536883', width=1)
        self._text(draw, (252, 137), 'WORKSPACE', 11, CYAN)
        centers, boxes = {}, {}
        self.hits = {}
        target = self.task.target_object_id if self.task else None
        active = self.goal.object_id if self.goal else None
        for index, group in enumerate(groups):
            cx = 30+(index+.5)*540/len(groups)
            draw.line((300, 157, cx, 169), fill='#41536c')
            self._text(draw, (cx-18, 167), region_name(group), 11, MUTED)
            keys = sorted(k for k, n in self.nodes.items() if n['parent'] == group)
            columns = 2 if len(keys) > 7 else 1
            for i, key in enumerate(keys[:14]):
                x = cx+(i % columns-(columns-1)/2)*38
                y = 194+(i//columns)*21
                box = (x-17, y-8, x+17, y+9)
                boxes[key], centers[key], self.hits[key] = box, (x, y), box
                draw.line((cx, 182, x, y), fill='#273b54')
            if len(keys) > 14:
                self._text(draw, (cx-12, 321), f'+{len(keys)-14}', 10, MUTED)
        pairs = self.explanation.get('support_pairs', [])
        chain = set(self.explanation.get('chain', [])) | {active, target, self.selected}
        for upper, lower in pairs:
            if upper in centers and lower in centers and (upper in chain or lower in chain):
                a, b = centers[upper], centers[lower]
                draw.line((a, b), fill=RED, width=2)
                draw.ellipse((b[0]-2, b[1]-2, b[0]+2, b[1]+2), fill=RED)
        for key, box in boxes.items():
            node = self.nodes[key]
            color = node['color']
            fill = tuple(int(np.clip(c, 0, 1)*150)+25 for c in color) if color is not None else '#314967'
            if node['status'] == 'history':
                fill = '#2a3546'
            border = GOLD if key == target else CYAN if key == active else GREEN if key in self.added else '#536883'
            draw.rounded_rectangle(box, radius=3, fill=fill, outline=border, width=2 if key in (target, active) else 1)
            if key == self.selected:
                draw.rectangle((box[0]-2, box[1]-2, box[2]+2, box[3]+2), outline=FG)
            self._text(draw, (box[0]+2, box[1]+1), short(key), 11, MUTED if node['status'] == 'history' else FG)
        selected = self.selected or active or target
        if selected in self.nodes:
            n = self.nodes[selected]
            position = ', '.join(f'{v:+.3f}' for v in n['pos'])
            self._text(draw, (30, 336), f'{selected} · {n["status"]} · {region_name(n["parent"])}  ({position}) m', 11)
        self._text(draw, (30, 355), '금색 목표  /  청록 실행 대상  /  회색 미관측  /  빨강 위→아래', 11, MUTED)

    def render(self, width, height):
        now = time.monotonic()
        if self._cache is None or now-self._cache_time >= .12:
            im = Image.new('RGB', (600, 960), BG)
            draw = ImageDraw.Draw(im)
            self._text(draw, (20, 12), 'A → B → C  LIVE INSPECTOR', 23)
            source = self.graph.source if self.graph else '-'
            t = self.observation.timestamp if self.observation else 0.
            state = '일시정지' if self.paused and not self.finished else self.status
            self._text(draw, (20, 47), f'{source}  ·  {self.controller.upper()}  ·  t={t:.2f}s', 14, MUTED)
            self._text(draw, (400, 47), state, 15, RED if self.error else GREEN if self.finished else CYAN)
            if self.task is not None:
                spec = self.task.target_spec
                if spec is not None:
                    color = tuple(int(c*255) for c in spec.reference_rgb)
                    draw.rectangle((20, 68, 34, 78), fill=color, outline=GOLD)
                target = '탐색 중' if self.task.target_object_id.startswith('unresolved:') else short(self.task.target_object_id)
                description = ((region_name(spec.region)+' ' if spec.region else '')+spec.category) if spec else 'ID 지정'
                self._text(draw, (42, 65), f'회수 목표 {target}  ·  {description}', 11, GOLD)
            self._graph(draw)
            self._card(draw, 389, 563)
            modes = dict(target_pick='목표 회수', unblock='방해물 제거', search='숨은 목표 탐색',
                         target_place='목표 배치', buffer_place='방해물 임시 배치')
            mode = modes.get(self.explanation.get('mode'), '계획 대기')
            self._text(draw, (30, 401), f'B  {mode}', 19)
            if self.goal:
                self._text(draw, (390, 405), f'#{self.index}  {self.goal.primitive} {short(self.goal.object_id)}', 14, GOLD)
                self._wrap(draw, (30, 434), self.explanation.get('reason', ''), lines=2)
                chain = self.explanation.get('chain', [])
                candidates = self.explanation.get('candidates', [])
                if len(chain) > 1:
                    self._text(draw, (30, 482), '현재 적층 (위→아래)  '+ ' → '.join(short(k) for k in reversed(chain)), 14, RED)
                elif candidates:
                    lines = [f'{short(c["object_id"])}: 숨음 {c["hidden"]}, 선반 합계 {c["shelf_hidden"]}'
                             if 'object_id' in c else c['label']+(' 비어 있음' if c['free'] else ' 점유됨') for c in candidates[:2]]
                    for i, line in enumerate(lines):
                        self._text(draw, (30, 480+i*19), line, 12, MUTED)
                relation = 'A 그래프의 적층 관계' if self.explanation.get('support_source') == 'graph' else 'B의 위치·크기 기반 적층 추정'
                self._text(draw, (30, 527), f'계획 t={self.plan_time:.2f}s  ·  {relation}', 12, MUTED)
            self._card(draw, 576, 745)
            chunk = f'{self.chunk_shape[0]}×{self.chunk_shape[1]}' if self.chunk_shape else '-'
            self._text(draw, (30, 589), f'C  현재 Subgoal / 실행 청크 {chunk}', 18)
            if self.goal:
                self._text(draw, (30, 620), f'{self.goal.primitive}({self.goal.object_id})', 15, CYAN)
                self._text(draw, (30, 646), '시작 물체 중심  '+xyz(self.goal.object_pose), 13)
                self._text(draw, (30, 668), '배치 중심  '+xyz(self.goal.place_pose), 13)
                obs = self.observation
                if obs is not None:
                    tactile = f'{int(obs.finger_contacts[0])} / {int(obs.finger_contacts[1])}' if obs.source == 'rgbd' else 'GT contact'
                    self._text(draw, (30, 692), f'그리퍼 {obs.robot_state[5]*100:.0f}% 닫힘  ·  접촉 {tactile}  ·  파지 {short(obs.held_object_id)}', 12, MUTED)
                draw.rounded_rectangle((30, 722, 570, 729), radius=3, fill='#2b3c54')
                draw.rounded_rectangle((30, 722, 30+max(2, 540*min(1, self.steps/max(1, self.budget))), 729), radius=3, fill=CYAN)
                self._text(draw, (445, 704), f'{self.steps} / {self.budget} tick', 11, MUTED)
                if self.command is not None:
                    self._text(draw, (30, 706), 'cmd  '+'  '.join(f'{v:+.2f}' for v in self.command), 10, MUTED)
            self._card(draw, 757, 843)
            self._text(draw, (30, 766), 'A  그래프 변경 이력 (이동 기준 5mm)', 14)
            for i, line in enumerate(list(self.changes)[:3]):
                self._text(draw, (30, 790+i*17), line, 11, MUTED)
            self._card(draw, 854, 925)
            self._text(draw, (30, 862), '실제로 완료한 Subgoal', 14)
            for i, (primitive, ident, ticks) in enumerate(self.history[-6:]):
                self._text(draw, (30+(i%2)*270, 887+(i//2)*12), f'✓ {primitive} {short(ident)} · {ticks} tick', 10, GREEN)
            if self.error:
                draw.rectangle((20, 849, 580, 924), fill='#442733')
                self._wrap(draw, (30, 858), self.error, size=14, lines=3, color=RED)
            self._text(draw, (20, 937), 'SPACE 재생/정지  ·  N 다음 Subgoal  ·  노드 클릭  ·  ESC 종료', 12, MUTED)
            self._cache, self._cache_time = im, now
        return np.asarray(self._cache.resize((width, height), Image.Resampling.BILINEAR)).copy()
