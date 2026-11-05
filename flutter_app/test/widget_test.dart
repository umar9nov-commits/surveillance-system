import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:surveillance_app/main.dart';

void main() {
  testWidgets('shows five groups and mode controls without patrol card', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1400, 1000));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(const MyApp());
    expect(find.text('Standing'), findsOneWidget);
    expect(find.text('Walking'), findsOneWidget);
    expect(find.text('Running'), findsOneWidget);
    expect(find.text('Start Simulation'), findsOneWidget);
    expect(find.text('1  Gazebo Simulation'), findsWidgets);
    expect(find.text('5  YOLO Activity Detection'), findsWidgets);
    expect(find.text('Robot Patrol'), findsNothing);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('mode selection does not create a camera detection', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1400, 1000));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(const MyApp());
    await tester.tap(find.text('Walking'));
    await tester.pump();
    expect(find.text('WALKING DETECTED'), findsNothing);
    await tester.pumpWidget(const SizedBox());
  });
}
