"""Five visible controls with real, separately routed process output."""
import argparse
import json
import queue
import signal
import sys
import threading
import time
from flutter_demo_runner import DemoHost
from flutter_demo_runtime import Cancelled

PREFIX = '@@SURVEILLANCE_COMPONENT@@'
GROUP = {
    'simulation': 'simulation', 'rosbridge': 'simulation',
    'localization': 'localization', 'navigation': 'navigation',
    'patrol': 'navigation', 'observation': 'navigation',
    'rviz': 'rviz', 'monitor': 'yolo', 'decision': 'yolo',
}
DEPENDENCIES = {
    'localization': ('simulation',),
    'navigation': ('simulation', 'localization'),
    'rviz': ('simulation',),
    'yolo': ('simulation', 'localization', 'navigation'),
}
STOP_GROUPS = {
    'yolo': ('observation', 'monitor', 'decision'),
    'navigation': ('observation', 'monitor', 'decision', 'patrol', 'navigation'),
    'localization': ('observation', 'monitor', 'decision', 'patrol', 'navigation', 'localization'),
    'rviz': ('rviz',),
}

class ModularHost(DemoHost):
    def __init__(self, mode):
        super().__init__(mode)
        self.states = {key: 'stopped' for key in ('simulation', 'localization', 'navigation', 'rviz', 'yolo')}
        self.reported = set()
        self.patrol_checked = False
        self.mission_checked = False
        self.command_queue = queue.Queue()
        self.reader_by_name = {}

    def start(self, name, command):
        super().start(name, command)
        self.reader_by_name[name] = self.readers[-1]

    def event(self, key, message='', state=None, line=None):
        if state is not None:
            self.states[key] = state
        print(PREFIX + json.dumps({
            'key': key, 'state': self.states[key],
            'message': message, 'line': line,
            'connection_ready': self.connection_ready,
        }), flush=True)

    def write(self, line):
        with self.lock:
            self.logfile.write(line + '\n')
            self.logfile.flush()
            key = 'simulation'
            if line.startswith('[') and ']' in line:
                key = GROUP.get(line[1:line.index(']')], key)
            try:
                self.event(key, line=line)
            except BrokenPipeError:
                self.stop.set()

    def stage(self, phase, message):
        key = 'navigation' if phase in (
            'navigation_startup', 'navigation_ready', 'standing_trigger',
            'observing', 'observation_reached', 'resuming_patrol', 'circuit_complete'
        ) else 'simulation'
        self.write('[' + key + '] ' + message)

    def wait_until(self, test, description, timeout=90):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            self.tick(tuple(name for name in ('simulation', 'rosbridge', 'localization', 'navigation')
                            if name in self.processes))
            if test():
                return
        raise RuntimeError('Timed out: ' + description)

    def open_world(self):
        self.preflight()
        self.event('simulation', 'Starting Gazebo and cameras...', 'starting')
        self.start('simulation', ['bash', str(self.root / 'start_flutter_world.sh'), '--' + self.mode, '--gui'])
        self.start('rosbridge', ['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'])
        self.wait_connection()
        self.wait_until(lambda: self.probe.clock > 0 and
                        time.monotonic() - self.probe.camera_arrival < 4,
                        'simulation clock and external camera', 150)
        self.event('simulation', 'Gazebo and cameras ready. Start AMCL Localisation next.', 'running')

    def start_component(self, key):
        if key not in DEPENDENCIES:
            raise ValueError('Unknown component: ' + key)
        if self.states[key] != 'stopped':
            return
        missing = [item for item in DEPENDENCIES[key] if self.states[item] != 'running']
        if missing:
            self.event(key, 'Start these first: ' + ', '.join(missing))
            return
        self.event(key, 'Starting...', 'starting')
        try:
            if key in ('localization', 'navigation', 'rviz'):
                self.start(key, ['/usr/bin/python3', '-u', str(self.root / 'flutter_components.py'), key])
            if key == 'localization':
                self.wait_until(self.localization_ready, 'active map server and AMCL')
            elif key == 'navigation':
                self.wait_navigation()
                self.enable_patrol()
                self.start('patrol', ['/usr/bin/python3', '-u', str(self.root / 'patrol_once.py')])
                self.wait_marker('patrol_goal', 25)
                self.check_checkpoint(complete=False)
                self.patrol_checked = False
                self.mission_checked = False
                self.event(key, 'Navigation ready. Patrol started automatically.')
            elif key == 'yolo':
                if self.mode == 'standing':
                    self.start('observation', ['/usr/bin/python3', '-u',
                               str(self.root / 'pose_observation_bridge.py'), '--execute'])
                    self.wait_marker('observation_armed', 20)
                    self.wait_node('pose_observation_bridge')
                self.start('decision', ['/usr/bin/python3', '-u', str(self.root / 'pose_decision_preview.py')])
                self.wait_marker('decision_ready', 20)
                self.wait_node('pose_decision_preview')
                self.start('monitor', [str(self.root / '.venv_yolo/bin/python'), '-u',
                                      str(self.root / 'live_pose_monitor.py'), '--seconds', '3600'])
                self.wait_marker('monitor_ready', 120)
                self.wait_publishers()
            elif key == 'rviz':
                self.wait_until(lambda: 'rviz' in self.probe.node.get_node_names(), 'RViz ROS node', 45)
            self.event(key, 'Ready.', 'running')
        except Exception:
            self.event(key, 'Startup failed. Stopping dependent components.', 'error')
            raise

    def localization_ready(self):
        names = self.probe.node.get_node_names()
        if any(names.count(name) != 1 for name in ('map_server', 'amcl', 'lifecycle_manager_localization')):
            return False
        return self.probe.map_received and self.probe.lifecycle_ready()

    def wait_marker(self, marker, timeout):
        self.wait_until(lambda: marker in self.markers, marker, timeout)

    def wait_node(self, name):
        self.wait_until(lambda: self.probe.node.get_node_names().count(name) == 1,
                        'exactly one ' + name, 15)

    def stop_names(self, names):
        if any(name in self.processes and name in names for name in
               ('patrol', 'navigation', 'localization', 'observation')):
            if self.controller_owned and self.probe:
                self.probe.enabled(False, allow_stopping=True)
        for name in names:
            if name not in self.processes:
                continue
            tree = self.trees[name]
            for sig, duration in ((signal.SIGINT, 10), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
                self.refresh()
                if not tree.alive():
                    break
                tree.send(sig)
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline and tree.alive():
                    self.refresh()
                    if self.probe:
                        self.probe.spin()
            if tree.alive():
                raise RuntimeError('Owned process remains: ' + name)
            self.processes[name].wait(timeout=2)
            # Join its reader before clearing errors/markers for a restart.
            self.reader_by_name.pop(name).join(timeout=2)
            del self.processes[name]
            del self.trees[name]
            self.errors.pop(name, None)

    def stop_component(self, key):
        names = STOP_GROUPS[key]
        if key == 'yolo':
            # A standing bridge can own a resumed patrol. Stop its navigation
            # group as well so no mission child can keep moving after Stop.
            names = STOP_GROUPS['navigation'] if 'observation' in self.processes else ('monitor', 'decision')
        affected = {GROUP[name] for name in names}
        for group in affected:
            self.event(group, 'Stopping...', 'stopping')
        self.stop_names(names)
        for group in affected:
            self.event(group, 'Stopped.', 'stopped')
        if key != 'rviz':
            self.markers.difference_update({
                'observation_armed', 'decision_ready', 'monitor_ready', 'standing_trigger',
                'observation_goal', 'observation_reached', 'patrol_resuming', 'mission_complete', 'no_resume'})
            self.reported.clear()
            self._handoff_error_seen = None
            self.mission_checked = False
            self.__dict__.pop('_observation_exit_seen', None)
        if 'navigation' not in self.processes:
            self.markers.difference_update({'patrol_goal', 'patrol_complete'})
            self.patrol_checked = False
            for attribute in ('_patrol_exit_seen', '_observation_exit_seen'):
                self.__dict__.pop(attribute, None)
            self.controller_owned = False

    def check_mission(self):
        for marker, text in (
            ('standing_trigger', '30 seconds accepted. Pausing patrol for observation.'),
            ('observation_goal', 'Observation goal sent.'),
            ('observation_reached', 'Observation point reached.'),
            ('patrol_resuming', 'Resuming the saved patrol leg.'),
        ):
            if marker in self.markers and marker not in self.reported:
                self.event('navigation', text)
                self.reported.add(marker)
        patrol = self.processes.get('patrol')
        if patrol is not None and patrol.poll() is not None and not self.patrol_checked:
            for thread in self.readers:
                if thread.is_alive():
                    thread.join(timeout=0.01)
            if self.mode == 'standing' and 'standing_trigger' in self.markers:
                self.patrol_checked = True
            elif 'patrol_complete' in self.markers:
                self.check_checkpoint(complete=True)
                self.event('navigation', 'Patrol circuit complete. Monitoring remains active.')
                self.patrol_checked = True
            else:
                first_seen = getattr(self, '_patrol_exit_seen', time.monotonic())
                self._patrol_exit_seen = first_seen
                if time.monotonic() - first_seen > 1:
                    raise RuntimeError('Patrol ended without completion or an observation handoff.')
        observation = self.processes.get('observation')
        if observation is not None and observation.poll() is not None and not self.mission_checked:
            first_seen = getattr(self, '_observation_exit_seen', time.monotonic())
            self._observation_exit_seen = first_seen
            if 'mission_complete' in self.markers:
                self.check_checkpoint(complete=True)
                self.event('navigation', 'Observation and resumed patrol circuit complete.')
                self.mission_checked = True
            elif 'no_resume' in self.markers:
                self.event('navigation', 'Observation completed after patrol had already finished.')
                self.mission_checked = True
            elif time.monotonic() - first_seen > 1:
                raise RuntimeError('Standing observation ended without confirmed completion.')

    def shutdown(self):
        if self.shutdown_started:
            return
        # Stop mission processes while simulator and controller services remain alive.
        try:
            self.stop_names(('observation', 'monitor', 'decision', 'patrol',
                             'rviz', 'navigation', 'localization'))
        finally:
            super().shutdown()

    def run(self):
        self.open_world()
        while not self.stop.is_set():
            required = tuple(name for name in ('simulation', 'rosbridge', 'localization',
                                                'navigation', 'monitor', 'decision')
                             if name in self.processes)
            self.tick(required)
            if time.monotonic() - self.probe.clock_arrival > 8:
                raise RuntimeError('Simulation clock stopped.')
            rviz = self.processes.get('rviz')
            if rviz is not None and rviz.poll() is not None:
                self.stop_component('rviz')
            self.check_mission()
            try:
                action, key = self.command_queue.get_nowait()
            except queue.Empty:
                continue
            if action == 'start':
                self.start_component(key)
            elif action == 'stop':
                self.stop_component(key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('standing', 'walking', 'running'))
    host = ModularHost(parser.parse_args().mode)
    def commands():
        for line in sys.stdin:
            try:
                data = json.loads(line)
                action, key = data['action'], data.get('key')
                if action == 'stop_all':
                    break
                if action in ('start', 'stop') and key in DEPENDENCIES:
                    host.command_queue.put((action, key))
            except (ValueError, KeyError, TypeError):
                host.write('[simulation] Invalid control command ignored.')
        host.stop.set()
    threading.Thread(target=commands, daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: host.stop.set())
    code = 0
    try:
        host.run()
    except Cancelled:
        pass
    except Exception as error:
        code = 1
        host.event('simulation', 'Session error: ' + str(error), 'error')
    finally:
        try:
            host.shutdown()
        except Exception as error:
            code = 1
            host.event('simulation', 'Shutdown needs attention: ' + str(error), 'error')
        host.logfile.close()
    return code

if __name__ == '__main__':
    raise SystemExit(main())
