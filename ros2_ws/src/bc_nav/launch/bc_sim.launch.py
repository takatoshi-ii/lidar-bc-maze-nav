"""模倣学習用のシミュレーション起動

  データ収集: ros2 launch bc_nav bc_sim.launch.py mode:=record
  自律走行:   ros2 launch bc_nav bc_sim.launch.py mode:=drive

ロボット・コース・ブリッジ設定は nav_dev のものを再利用する。
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    nav_dev_dir = get_package_share_directory('nav_dev')
    model_path = os.path.join(nav_dev_dir, 'models')

    mode = LaunchConfiguration('mode')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    yaw = LaunchConfiguration('yaw')
    is_record = IfCondition(PythonExpression(["'", mode, "' == 'record'"]))
    is_drive = IfCondition(PythonExpression(["'", mode, "' == 'drive'"]))

    #ignition gazeboがモデルにアクセスできるように設定
    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[os.path.join('/opt/ros/humble', 'share') + ':' + model_path])

    ign_gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('ros_ign_gazebo'),
                          'launch', 'ign_gazebo.launch.py')]),
        launch_arguments=[('ign_args', [' -r -v 3 ' +
                          os.path.join(get_package_share_directory('bc_nav'),
                                       'worlds', 'bc_world.sdf')])])

    spawn_robot = Node(
        package='ros_ign_gazebo', executable='create', output='screen',
        arguments=['-name', 'LidarRobo',
                   '-file', os.path.join(model_path, 'LidarRobo', 'model.sdf'),
                   '-allow_renaming', 'true',
                   '-x', x, '-y', y, '-z', '0.075', '-Y', yaw])

    spawn_field = Node(
        package='ros_ign_gazebo', executable='create', output='screen',
        arguments=['-file', os.path.join(model_path, 'field', 'model.sdf'),
                   '-allow_renaming', 'false'])

    bridge = Node(
        package='ros_ign_bridge', executable='parameter_bridge', output='screen',
        parameters=[{'config_file': os.path.join(nav_dev_dir, 'config', 'nav_slam.yaml')},
                    {'use_sim_time': True}])

    #データ収集モード: 人がテレオペし、その操作を記録する
    teleop_node = Node(
        package='bc_nav', executable='bc_teleop', output='screen',
        prefix='xterm -e', condition=is_record)
    recorder = Node(
        package='bc_nav', executable='data_recorder', output='screen',
        condition=is_record)

    #自律走行モード: 学習済みモデルが操作する
    driver = Node(
        package='bc_nav', executable='bc_driver', output='screen',
        parameters=[{'model_path': LaunchConfiguration('model_path'),
                     #180 のような整数で渡されても小数として扱う
                     'duration': ParameterValue(LaunchConfiguration('duration'), value_type=float),
                     'start_x': ParameterValue(x, value_type=float),
                     'start_y': ParameterValue(y, value_type=float)}],
        condition=is_drive)

    #動画撮影: 真上カメラの映像に情報を重ねて ~/bc_nav_data/videos に保存する
    record_video = IfCondition(LaunchConfiguration('record_video'))
    camera_bridge = Node(
        package='ros_ign_bridge', executable='parameter_bridge', name='camera_bridge',
        output='screen',
        parameters=[{'config_file': os.path.join(get_package_share_directory('bc_nav'),
                                                 'config', 'bc_camera_bridge.yaml')}],
        condition=record_video)
    video_recorder = Node(
        package='bc_nav', executable='video_recorder', output='screen',
        parameters=[{
            'prefix': mode,
            'title': PythonExpression(["'Human demonstration' if '", mode, "' == 'record' "
                                       "else 'AI driving (Behavior Cloning)'"]),
            'actor': PythonExpression(["'HUMAN' if '", mode, "' == 'record' else 'AI'"]),
        }],
        condition=record_video)

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='record', description='record または drive'),
        DeclareLaunchArgument('model_path',
                              default_value=os.path.expanduser('~/bc_nav_data/model.pt')),
        DeclareLaunchArgument('duration', default_value='0.0',
                              description='自律走行を自動停止するまでの秒数(0なら無制限)'),
        DeclareLaunchArgument('record_video', default_value='false',
                              description='true で真上カメラの映像をMP4に保存する'),
        DeclareLaunchArgument('x', default_value='0.4'),
        DeclareLaunchArgument('y', default_value='0.4'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        ign_resource_path,
        ign_gz,
        spawn_field,
        spawn_robot,
        bridge,
        teleop_node,
        recorder,
        driver,
        camera_bridge,
        video_recorder,
    ])
