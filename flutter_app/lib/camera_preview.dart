import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';

class CameraPreviewPanel extends StatefulWidget {
  const CameraPreviewPanel({super.key, required this.enabled});
  final bool enabled;

  @override
  State<CameraPreviewPanel> createState() => _CameraPreviewPanelState();
}

class _CameraPreviewPanelState extends State<CameraPreviewPanel> {
  static const _prefix = '@@SURVEILLANCE_CAMERA@@';
  static const _command =
      'exec bash ~/surveillance_ws/camera_preview_entry.sh --source ';

  Process? _process;
  Timer? _timer;
  Uint8List? _image;
  DateTime? _received;
  String _source = 'external';
  String _message = 'Waiting for the simulation connection.';
  int _generation = 0;

  bool get _fresh =>
      _received != null && DateTime.now().difference(_received!).inSeconds < 5;

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {});
      }
    });
    if (widget.enabled) {
      unawaited(_restart());
    }
  }

  @override
  void didUpdateWidget(covariant CameraPreviewPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.enabled != widget.enabled) {
      unawaited(_restart());
    }
  }

  Future<void> _finish(Process process) async {
    try {
      process.stdin.writeln('stop');
      await process.stdin.flush();
      await process.stdin.close();
    } catch (_) {
      // The preview may already have exited.
    }
    try {
      await process.exitCode.timeout(const Duration(seconds: 5));
    } on TimeoutException {
      // Closing the WSL client's pipes also closes the helper's stdin.
      process.kill();
    }
  }

  Future<void> _restart() async {
    final generation = ++_generation;
    final previous = _process;
    _process = null;
    if (mounted) {
      setState(() {
        _image = null;
        _received = null;
        _message = widget.enabled
            ? 'Opening camera preview...'
            : 'Waiting for the simulation connection.';
      });
    }
    if (previous != null) {
      await _finish(previous);
    }
    if (!mounted || generation != _generation || !widget.enabled) {
      return;
    }
    final source = _source;
    try {
      final process = await Process.start('wsl.exe', [
        '-d',
        'Ubuntu',
        '--exec',
        'bash',
        '-lc',
        '$_command$source',
      ]);
      if (!mounted || generation != _generation || !widget.enabled) {
        // Drain output until the process closes, including startup messages.
        unawaited(process.stdout.drain<void>());
        unawaited(process.stderr.drain<void>());
        await _finish(process);
        return;
      }
      _process = process;
      process.stdout
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(
            (line) {
              if (!mounted ||
                  generation != _generation ||
                  !line.startsWith(_prefix)) {
                return;
              }
              try {
                final data = jsonDecode(line.substring(_prefix.length));
                if (data is! Map ||
                    data['kind'] != 'frame' ||
                    data['source'] != source) {
                  return;
                }
                final encoded = data['jpeg'];
                if (encoded is! String || encoded.length > 512000) {
                  return;
                }
                final bytes = base64Decode(encoded);
                if (bytes.isEmpty) {
                  return;
                }
                setState(() {
                  _image = bytes;
                  _received = DateTime.now();
                  _message = 'Waiting for a fresh camera frame.';
                });
              } catch (_) {
                // Ignore incomplete preview output; the next frame can recover.
              }
            },
            onError: (Object error) {
              debugPrint('Camera preview output: $error');
            },
          );
      process.stderr
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(
            (line) => debugPrint('[camera preview] $line'),
            onError: (Object error) => debugPrint('Camera preview: $error'),
          );
      unawaited(
        process.exitCode.then((code) {
          if (mounted && generation == _generation) {
            setState(() {
              _process = null;
              _image = null;
              _received = null;
              _message =
                  'Preview stopped. Check that the simulation is running, '
                  'then select Retry.';
            });
          }
        }),
      );
      if (mounted &&
          generation == _generation &&
          identical(_process, process)) {
        setState(() => _message = 'Waiting for camera frames...');
      }
    } catch (error) {
      if (mounted && generation == _generation) {
        setState(
          () => _message = 'Could not open the camera preview. Select Retry.',
        );
      }
      debugPrint('Camera preview start: $error');
    }
  }

  @override
  void dispose() {
    ++_generation;
    _timer?.cancel();
    final process = _process;
    _process = null;
    if (process != null) {
      unawaited(_finish(process));
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 12,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Text(
                'Camera preview',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              for (final entry in const [
                ('external', 'External camera'),
                ('robot', 'Robot camera'),
              ])
                ChoiceChip(
                  label: Text(entry.$2),
                  selected: _source == entry.$1,
                  onSelected: (_) {
                    if (_source != entry.$1) {
                      setState(() => _source = entry.$1);
                      unawaited(_restart());
                    }
                  },
                ),
              Text(
                _fresh && widget.enabled ? 'LIVE' : 'WAITING',
                style: TextStyle(
                  color: _fresh && widget.enabled
                      ? Colors.tealAccent
                      : Colors.white54,
                ),
              ),
              TextButton.icon(
                onPressed: widget.enabled ? () => unawaited(_restart()) : null,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Expanded(
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                color: const Color(0xFF080E16),
                border: Border.all(color: Colors.white12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: _image != null && _fresh && widget.enabled
                  ? Image.memory(
                      _image!,
                      fit: BoxFit.contain,
                      gaplessPlayback: true,
                      errorBuilder: (_, error, stack) => const Center(
                        child: Text('Waiting for a valid camera frame.'),
                      ),
                    )
                  : Center(
                      child: Text(
                        widget.enabled ? _message : 'Start the simulation and connect to ROS to view a camera.',
                      ),
                    ),
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Preview runs while this tab is open.',
            style: TextStyle(color: Colors.white54),
          ),
        ],
      ),
    );
  }
}
