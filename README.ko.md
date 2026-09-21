# RL_communication

[English](README.md)

여러 에이전트가 협력해서 격자 위의 직사각형 물체를 목표 지점까지 밀어 옮기는
멀티에이전트 강화학습 샌드박스. PPO로 단계별 커리큘럼(stage 0 → 7)을 거쳐
학습시키고, out-of-distribution 벤치마크로 검증한다.

## 개요

- **환경**: `CooperativeTransportEnv` (`env/cooperative_transport_env.py`)는
  고정 크기 격자 위에서 동작하는 [PettingZoo](https://pettingzoo.farley.ml/)
  `ParallelEnv`다. 에이전트들은 `target` 물체를 `goal_cells`까지 밀어야 하며,
  경우에 따라 `obstacle` 물체를 피해가야 한다. 물체를 미는 것은 해당 면에
  접촉해야 하는 모든 에이전트가 같은 스텝에 같은 방향을 선택했을 때만
  성공한다 (`_resolve_movement` 참고).
- **관측(Observation)**은 에이전트별로 부분 관측 가능(partially observable)하다:
  로컬 `vision_size × vision_size` 윈도우(`local_map`, `object_ids`,
  `goal_mask`), 에이전트의 전역 `position`, 물체별 접촉 정보
  `touching_objects`, 그리고 다른 에이전트가 보낸 상대 위치를 담는
  `messages` 채널로 구성된다.
- **액션(Action)**은 에이전트당 `MultiDiscrete([5, 2])`다: 이동 액션
  (`STAY/UP/RIGHT/DOWN/LEFT`)과 통신 액션(`NO_MESSAGE`/`HELP`)으로 나뉜다.
  에이전트가 `HELP`를 보내면 다른 모든 에이전트는 다음 관측의 `messages`
  필드에서 그 에이전트의 상대 좌표를 받는다. 다만 현재 커리큘럼 학습
  스크립트들은 항상 `NO_MESSAGE`만 보낸다 — 통신 채널은 환경에 구현되어
  있고 관측 가능하지만, 아직 실제로 학습에 사용되지는 않는다.
- **학습**: `train/ppo.py`는 표준적인 clipped-objective PPO이며, 공유
  MLP `ActorCritic`(`train/network.py`)과 GAE 기반 rollout
  buffer(`train/rollout_buffer.py`)를 사용한다. 모든 에이전트가 하나의
  정책을 공유한다.
- **커리큘럼**: 점점 어려워지는 시나리오를 순서대로 학습한다
  (`train/stages/stage_0_6` → `train/stages/stage_7`). 이동 평균 성공률이
  기준치를 넘으면 자동으로 다음 단계로 진행한다.
- **평가**: `evaluation/`은 고정된(freeze) 체크포인트를 in-distribution 및
  out-of-distribution 벤치마크 시나리오(위치 이동/장애물/목표 변형, 복합
  시나리오, 스트레스 테스트)에 대해 실행해서 CSV/JSON 요약과, 필요하면
  시각화용 에피소드 궤적(trajectory)을 생성한다.

## 프로젝트 구조

```
env/                      CooperativeTransportEnv, RectObject, 상수
train/
  network.py               ActorCritic (공유 policy/value MLP)
  ppo.py                    PPO 업데이트 루프, act/value, save/load
  rollout_buffer.py         멀티에이전트 GAE rollout buffer
  stages/
    common.py               모든 stage가 공유하는 set_seed, flatten_obs
    stage_0_6/               Stage 0-6 커리큘럼 매니저 + 학습 스크립트
    stage_7/                 Stage 7(차선 이동 일반화) 커리큘럼, 학습
                             스크립트, 왼쪽 차선 fine-tuning, 차선별
                             진단 스크립트
  evaluate.py               단일 체크포인트용 임시 평가 헬퍼
evaluation/
  benchmark_scenarios.py    IID / 장애물 / target-goal / 복합 / 스트레스
                             시나리오 정의
  evaluator.py              PolicyEvaluator: 에피소드 실행, 지표 집계
  run_baseline_benchmark.py 전체 벤치마크 실행용 CLI 진입점
  trajectory.py             에피소드 궤적 기록
  visualize.py               궤적 시각화(예: GIF로 렌더링)
checkpoints/               stage/마일스톤별 저장된 PPO 체크포인트
evaluation_results/         CSV/JSON 벤치마크 결과, 기록된 궤적
runs/                      TensorBoard 이벤트 로그
logs/                      텍스트 학습 로그
tests/test_env.py          PettingZoo API 준수 여부 + 환경 단위 테스트
main.py                    랜덤 정책 스모크 테스트 / 렌더링 데모
```

## 설치

```bash
pip install -r requirements.txt
```

Python 3.12 호환 `numpy`, `gymnasium`, `pettingzoo`, `torch`, `tensorboard`,
`matplotlib`, `pillow`가 필요하며, 테스트를 위해 `pytest`도 필요하다.

## 실행 방법

**랜덤 액션으로 환경 스모크 테스트 (콘솔에 렌더링):**

```bash
python main.py
```

**환경 단위 테스트 실행:**

```bash
pytest tests/test_env.py
```

**Stage 0-6 커리큘럼 처음부터 학습:**

```bash
python -m train.stages.stage_0_6.train
```

TensorBoard 로그는 `runs/curriculum_v2`에, 주기적 체크포인트는
`checkpoints/episode_*.pt`에, stage별 mastery 체크포인트는
`stage_N_mastered.pt`로 저장된다.

**Stage 6 mastery 체크포인트에서 이어서 Stage 7(차선 이동 일반화) 학습:**

```bash
python -m train.stages.stage_7.train
```

`checkpoints/stage_6_mastered.pt`를 불러온 뒤 target/goal/obstacle의 차선
위치를 점진적으로 이동시킨다(`stage_7A_shift1` → `stage_7B_shift2` →
`stage_7C_all_lanes`). 각 substage의 성공률을 독립적으로 추적한다.

**차선별 취약점 진단 / 취약 차선 fine-tuning:**

```bash
python -m train.stages.stage_7.lane_diagnostic
python -m train.stages.stage_7.finetune_left_lanes
```

**체크포인트를 IID + OOD 시나리오 세트로 벤치마킹:**

```bash
python -m evaluation.run_baseline_benchmark --checkpoint checkpoints/stage_7_mastered.pt --mode deterministic
```

결과는 `evaluation_results/<mode>/`에 저장된다 (`baseline_episodes.csv`,
`baseline_summary.json`, 그리고 대표적인 성공/실패 사례의 기록된 궤적).

## 커리큘럼 단계 (0-6)

각 stage는 스폰 분포를 넓히거나 장애물을 추가한다.
`CurriculumManager`(`train/stages/stage_0_6/curriculum.py`)는 최근 100
에피소드 이동 평균 성공률이 0.85를 넘으면(그리고 stage별 최소 에피소드
수를 채우면) 다음 stage로 진행시킨다:

| Stage | 변화 내용 |
|---|---|
| 0 | 고정된 에이전트/target/goal 배치, 장애물 없음 |
| 1-2 | 에이전트가 target 근처에 랜덤 스폰; target/goal 영역 확대 |
| 3 | 스폰 반경이 sub-level에 따라 확대(반경 3 → 5 → 완전 랜덤) |
| 4 | 첫 장애물 등장; target-only 보상만으로 mastery에 실패할 경우에만 장애물을 통로 밖으로 미는 shaping 보상 활성화 |
| 5 | 장애물이 있는 상태에서 완전 랜덤 스폰 |
| 6 | 두 번째 장애물 추가 (기본 커리큘럼의 마지막 stage) |

이후 Stage 7(`train/stages/stage_7/curriculum.py`)은 Stage 6의 전체 배치를
맵의 여러 차선(lane)으로 좌우 이동시켜서 일반화 성능을 검증한다.

## 참고 사항

- 물체를 밀려면 그 면에 접촉해야 하는 모든 에이전트가 같은 스텝에 같은
  방향을 선택해야 한다. 충돌하는 push/이동은
  `env/cooperative_transport_env.py`의 `_resolve_agent_conflicts` /
  `_remove_object_conflicts`에서 해소된다.
- 보상 = step penalty + 목표까지의 거리 기반 shaping (+ `obstacle_shaping`
  모드에서는 장애물을 통로 밖으로 미는 shaping 보상 추가) + 종료 시 성공
  보너스이며, 모든 에이전트에게 동일하게 공유된다.
- `env.state()`는 평탄화된 전역 격자를 반환한다(centralized critic이나
  디버깅에 활용 가능하지만, 현재 PPO 구성에서는 에이전트별 관측만
  사용한다).
