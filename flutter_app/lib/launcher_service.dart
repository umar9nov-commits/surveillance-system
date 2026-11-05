import 'dart:async';
import 'dart:convert';
import 'dart:io';

class LauncherService {
  LauncherService(this.onChanged);
  final void Function() onChanged;
  static const keys = [
    'simulation',
    'localization',
    'navigation',
    'rviz',
    'yolo',
  ];
  static const titles = {
    'simulation': 'Gazebo Simulation',
    'localization': 'AMCL Localisation',
    'navigation': 'Custom Navigation',
    'rviz': 'RViz',
    'yolo': 'YOLO Activity Detection',
  };
  final Map<String, String> states = {for (final key in keys) key: 'stopped'};
  final Map<String, List<String>> consoles = {for (final key in keys) key: []};
  Process? _process;
  bool _starting = false;
  bool _stopping = false;
  bool _disposed = false;
  bool connectionReady = false;
  String selectedConsole = 'simulation';
  String phase = 'Choose a mode, then start Gazebo Simulation.';

  bool running(String key) =>
      states[key] == 'running' || states[key] == 'starting';
  bool busy(String key) =>
      states[key] == 'starting' ||
      states[key] == 'stopping' ||
      _starting ||
      _stopping;
  bool get hasProcesses => _process != null || _starting;
  List<String> get logs => consoles[selectedConsole]!;
  bool get changing =>
      _starting ||
      _stopping ||
      states.values.any((state) => state == 'starting' || state == 'stopping');

  bool canStart(String key) {
    if (changing || states[key] != 'stopped') return false;
    if (key == 'simulation') return !hasProcesses;
    if (_process == null || states['simulation'] != 'running') return false;
    if (key == 'navigation') return states['localization'] == 'running';
    if (key == 'yolo') return states['navigation'] == 'running';
    return true;
  }

  void _changed() {
    if (!_disposed) onChanged();
  }

  void selectConsole(String key) {
    selectedConsole = key;
    _changed();
  }

  void _note(String key, String message) {
    final clock = DateTime.now().toIso8601String().substring(11, 19);
    final lines = consoles[key]!;
    lines.add('$clock  $message');
    if (lines.length > 500) lines.removeRange(0, lines.length - 500);
    _changed();
  }

  void _receive(String line) {
    const prefix = '@@SURVEILLANCE_COMPONENT@@';
    if (line.startsWith(prefix)) {
      try {
        final event = jsonDecode(line.substring(prefix.length));
        if (event is Map && keys.contains(event['key'])) {
          final key = event['key'] as String;
          states[key] = event['state'] as String;
          connectionReady = event['connection_ready'] == true;
          final message = event['message'];
          final output = event['line'];
          if (message is String && message.isNotEmpty) {
            phase = message;
            _note(key, message);
          } else if (output is String) {
            _note(key, output);
          } else {
            _changed();
          }
          return;
        }
      } catch (_) {
        /* Keep malformed output visible. */
      }
    }
    _note('simulation', line);
  }

  Future<void> start(String key, String mode) async {
    if (_disposed || !canStart(key)) return;
    selectConsole(key);
    if (key != 'simulation') {
      states[key] = 'starting';
      _changed();
      await _send({'action': 'start', 'key': key});
      return;
    }
    if (!['standing', 'walking', 'running'].contains(mode)) return;
    _starting = true;
    states[key] = 'starting';
    connectionReady = false;
    phase = 'Starting $mode simulation...';
    _note(key, phase);
    try {
      final command =
          'source /opt/ros/jazzy/setup.bash && '
          'source /home/umaryasir/surveillance_ws/install/setup.bash && '
          'exec /usr/bin/python3 -u '
          '/home/umaryasir/surveillance_ws/flutter_modular_runner.py --mode $mode';
      final process = await Process.start('wsl.exe', [
        '-d',
        'Ubuntu',
        '--exec',
        'bash',
        '-lc',
        command,
      ]);
      _process = process;
      final readers = <Future<void>>[];
      for (final stream in [process.stdout, process.stderr]) {
        readers.add(
          stream
              .transform(const Utf8Decoder(allowMalformed: true))
              .transform(const LineSplitter())
              .forEach(_receive),
        );
      }
      unawaited(
        process.exitCode.then((code) async {
          await Future.wait(readers);
          if (!identical(_process, process)) return;
          _process = null;
          _stopping = false;
          connectionReady = false;
          for (final key in keys) {
            states[key] = 'stopped';
          }
          phase = code == 0
              ? 'Simulation stopped.'
              : 'Session ended with an error. Read the consoles.';
          _note('simulation', '$phase Exit code: $code.');
        }),
      );
      if (_disposed) {
        process.stdin.writeln(jsonEncode({'action': 'stop_all'}));
        unawaited(process.stdin.close());
      }
    } catch (error) {
      states[key] = 'stopped';
      _note(key, 'Unable to start: $error');
    } finally {
      _starting = false;
      _changed();
    }
  }

  Future<void> _send(Map<String, String> command) async {
    try {
      final process = _process;
      if (process == null) throw StateError('No active simulation session.');
      process.stdin.writeln(jsonEncode(command));
      await process.stdin.flush();
    } catch (error) {
      final key = command['key'] ?? 'simulation';
      states[key] = 'error';
      _note(key, 'Command failed: $error');
    }
  }

  Future<void> stop(String key) async {
    if (_process == null || _stopping) return;
    if (key == 'simulation') {
      _stopping = true;
      for (final entry in keys) {
        if (states[entry] != 'stopped') states[entry] = 'stopping';
      }
      _changed();
      await _send({'action': 'stop_all'});
    } else {
      states[key] = 'stopping';
      _changed();
      await _send({'action': 'stop', 'key': key});
    }
  }

  void clearLogs() {
    consoles[selectedConsole]!.clear();
    _changed();
  }

  void dispose() {
    _disposed = true;
    final process = _process;
    if (process != null) {
      process.stdin.writeln(jsonEncode({'action': 'stop_all'}));
      unawaited(process.stdin.close());
    }
  }
}
