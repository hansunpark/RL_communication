# 통신 협동 운반 커리큘럼

> **목적.** Stage 0-7은 사실상 고정된 배치(target (4,2), goal은 그 바로 아래)에서만 검증됐다.
> 이를 넘어 (a) target/goal/obstacle의 크기·위치·개수와 스폰 위치가 모두 랜덤인 조건으로
> 일반화하고, (b) 통신(HELP 액션)이 실제로 학습되어 성능에 기여하게 만든다.
>
> **범위 (2026-09-17 합의).** 맵 크기 12x12, 에이전트 4명은 고정. 나머지 기하학 요소는 모두
> 랜덤화 대상. 통신은 "있으면 좋은" 기능이 아니라 학습되고 기여해야 하는 목표다.

---

## 0. 자율 진행 모드 (2026-10-20부터)

사용자가 워크스테이션을 직접 다룰 수 없어서, 크레딧이 남는 동안 Claude가 스스로 진행한다.
받은 지시:

1. GRU(RNN) 모델로 Track B 끝(S10a1 통과)에서 이어서 단계별 학습을 계속한다. GPU 최적화에
   몰두하지 않는다 (잘 안 되면 CPU).
2. 통신 on/off 평가에 **성공률, RC, CPR**을 함께 쓴다 (정의는 3장 "협동 지표").
3. 정체하면 커리큘럼을 쪼개거나 바꿔서 스스로 풀어 본다.
4. 끝나기 전에 무슨 일이 있었는지 기록한다 (아래 "자율 진행 기록").

운영 방식:

- 학습은 `Win32_Process.Create`(WMI)로 띄워 Claude 세션이 끝나도 계속 돈다. 띄우는 명령은
  "자율 진행 기록"에 그대로 남긴다.
- 푸시는 이 워크스테이션에 GitHub 인증이 없어서 안 된다. 커밋은 로컬 브랜치
  `comm-curricula-gru`에 쌓인다 (`git push -u origin comm-curricula-gru`를 사람이 한 번 실행해야 함).

### 자율 진행 기록

| 시각 | 내용 |
|---|---|
| 2026-10-20 | GRU 준비 완료, 커밋 `b6e7e7f`(코드), `4d89d9d`(결과물). 푸시 실패(인증 없음). |
| 2026-10-20 | env worker 병렬화(`EnvPool`)와 GPU 큰 batch 옵션 추가. GRU 기준 처리량: env 16개 CPU 1,626 → env 384개, worker 22개, GPU 14,374 env-step/s. RC/CPR 지표 추가. 커밋 `635c7c8`. |
| 2026-10-20 | **Track B GRU 학습 시작** (`comm_scratch_gru`): S10a1 통과 MLP 체크포인트에서 S10a2(레벨 15)부터. 명령은 아래. |
| 2026-10-20 | MLP 체크포인트 RC/CPR/모이는 속도 평가 (6장). 통신은 "모이는 속도"를 바꾼다 (S9: 22.7 / 68.5 step). |
| 2026-10-20 | **GRU가 S10a2를 70,130 episode에 통과.** 같은 체크포인트에서 시작한 MLP는 53,800 episode 동안 큰 target 0-4%였다 (batch 설정도 다르므로 GRU만의 효과로 단정하지는 않는다). ablation: 성공률 0.83 / 0.57, RC 0.56 / 0.44, **CPR 0.59 / 0.50** — MLP와 달리 CPR에도 통신 차이가 보인다. |
| 2026-10-20 | **GRU가 S10b(goal 무제한)를 70,205 episode에 통과.** MLP는 같은 레벨에서 45,000 episode 동안 최저 버킷 42-64%로 정체했었다. ablation: 성공률 0.90 / 0.59 (+0.30), RC 0.56 / 0.49, CPR 0.65 / 0.60. |
| 2026-10-21 | GRU가 S10c(스폰 완전 랜덤 + 작은 obstacle)를 40,628 episode에 통과. ablation: 성공률 0.89 / 0.57 (+0.32), RC 0.50 / 0.45, CPR 0.68 / 0.63. |
| 2026-10-21 | **GRU S10d(보통 크기 obstacle 1개) 정체**: 200,000 episode에 최저 버킷 74-84%를 오르내림. 가장 약한 "큰 target + goal 9칸 이상"은 obstacle 경로 위 58%, 큰 obstacle 62% (경로 밖 76%, 작은 obstacle 73%). 정체 ablation: 성공률 0.86 / 0.56, RC 0.46 / 0.44, CPR 0.72 / 0.63. 로그 `logs/comm_scratch_gru_train_part1_S10a2_to_S10d.log`. |
| 2026-10-21 | 대응: S10a2와 S10b 사이에 **S10a3**(보통 크기 obstacle이 경로를 막는 발판: 같은 행/열 goal 3-5칸, 근처 스폰, max_steps 100, entropy 0.03) 추가. S10d 정체 체크포인트에서 S10a3(레벨 16)부터 재시작 → S10b, S10c, S10d를 다시 거친다. 같은 설정, 명령은 `--start-level 16 --start-checkpoint checkpoints\comm_scratch_gru_S10d_one_obstacle_random_spawn_stalled.pt`. |
| 2026-10-21 | 첫 GRU 실행이 끝난 뒤 env worker 하나가 남아 있어서 `EnvPool.close()`가 정상 종료하지 않은 worker를 terminate하도록 고쳤다. |
| 2026-10-21 | S10a3가 50,000 episode에 버킷 68-76%로 느렸다. obstacle 모양별로 1x1 90%, 2x1 86%, 1x2/3x1 67%, **2x2 25%, 3x2 21%**, **벽에 붙은 obstacle 16%**. 2칸 두께 obstacle은 2-3명이 옆면에서 함께 밀어야 하고, 벽에 붙은 것은 벽 반대쪽으로 밀 수 없다. → S10a3을 **S10a3a**(면적 3 이하 얇은 obstacle) / **S10a3b**(보통 크기 전체)로 쪼개고, 두 발판 모두 obstacle을 벽에서 떨어뜨렸다. S10a3 50k 체크포인트(`comm_scratch_gru_S10a3_50k.pt`)에서 S10a3a(레벨 16)부터 재시작. 로그 `logs/comm_scratch_gru_train_part2_S10a3.log`. |
| 2026-10-21 | **S10a3a 11,723 episode, S10a3b 27,292 episode에 통과.** ablation S10a3a: 성공률 0.82 / 0.50, RC 0.46 / 0.39, **CPR 0.75 / 0.62 (+0.13, 지금까지 가장 큰 CPR 차이)**. S10a3b: 성공률 0.77 / 0.52, RC 0.48 / 0.38, CPR 0.72 / 0.64. obstacle을 치운 뒤 target을 미는 단계적 작업에서 통신이 동기화에도 쓰이는 것으로 보인다. 이어서 S10b, S10c, S10d를 다시 거친다. |
| 2026-10-21 | S10b 13,434 episode (처음 70,205), S10c 7,312 episode (처음 40,628)에 다시 통과. ablation S10b: 0.93 / 0.58, RC 0.54 / 0.44, CPR 0.68 / 0.59. S10c: 0.91 / 0.56, RC 0.47 / 0.44, CPR 0.71 / 0.62. |
| 2026-10-21 | **S10d를 60,093 episode에 통과** (발판 추가 전 200,000 episode 정체). ablation: 성공률 0.88 / 0.50 (+0.38), RC 0.45 / 0.41, CPR 0.71 / 0.71. 다음은 S10(split 1near + obstacle 0-2개), S11(obstacle 0-4개). |
| 2026-10-21 | S10이 100,000 episode에 16개 버킷 중 11개, 최저 61-65%로 거의 정체. 가장 약한 "큰 target + goal 9칸 이상 + near1"에서 obstacle 1개 77%, **2개 45%**, 경로 위 50%. S10은 S10d 대비 스폰(랜덤 → split 1near)과 obstacle 수(1 → 0-2) 두 축을 함께 바꾼다. → S10d와 S10 사이에 **S10e**(랜덤 스폰 + obstacle 0-2개)를 넣고 S10 100k 체크포인트(`comm_scratch_gru_S10_100k.pt`)에서 S10e(레벨 21)부터 재시작. 로그 `logs/comm_scratch_gru_train_part3_S10a3a_to_S10.log`. |
| 2026-10-21 | S10e 20,000 episode: 16개 버킷 중 11개, 최저 "큰 target + goal 9칸 이상 + obstacle" 58% (S10과 같은 곳). 진단 두 가지: (1) **시간 제한이 아니다** — max_steps를 200 → 400으로 늘려도 큰 target + 먼 goal + obstacle 2개 성공 102/200 그대로. (2) **풀 수 없는 배치도 아니다** — 에이전트를 무시하고 물체 배치만으로 BFS 해 보니 S10d/S10e 샘플 600개 모두 풀 수 있었다. 남은 원인은 obstacle 두 개를 차례로 처리하는 기술 부족. 100,000 episode까지 보고 정체하면 obstacle 2개 발판을 넣는다. |
| 2026-10-21 | S10e 100,000 episode: 12/16, 최저 66%로 정체. → S10d와 S10e 사이에 **S10f**(경로 막는 보통 크기 obstacle + 다른 곳 obstacle 1개, 같은 행/열 goal 4-8칸, 근처 스폰, max_steps 150, entropy 기본값) 추가. S10e 100k 체크포인트(`comm_scratch_gru_S10e_100k.pt`)에서 S10f(레벨 21)부터 재시작. 로그 `logs/comm_scratch_gru_train_part4_S10e.log`. |
| 2026-10-21 | S10f가 50,000 episode에 0/4 버킷 (goal 6-8칸 41-46%). 진단: 발판의 max_steps를 150으로 줄인 것이 실수였다 — 성공 episode 상위 10%가 180 step쯤이고, 같은 체크포인트로 150 step이면 6-8칸 53%, 300 step이면 67%. 발판이 실제 레벨보다 어려워졌다. → S10f max_steps를 기본값(200)으로 되돌리고 S10f 50k 체크포인트(`comm_scratch_gru_S10f_50k_150steps.pt`)에서 재시작. 로그 `logs/comm_scratch_gru_train_part5_S10f_150steps.log`. |
| 2026-10-21 | S10f(200 step)도 50,000 episode(합계 약 100,000)에 0/4, 최저 52-68%. S10a3 때처럼 앞에 더 쉬운 **S10f1**(goal 3-5칸, 얇은 경로 obstacle, 작은 추가 obstacle 1x1-2x1)을 두고 S10f 100k 체크포인트(`comm_scratch_gru_S10f_100k.pt`)에서 S10f1(레벨 21)부터 재시작. 로그 `logs/comm_scratch_gru_train_part6_S10f.log`. |
| 2026-10-21 | S10f1은 1,420 episode 만에 통과 (ablation: 성공률 0.87 / 0.64, RC 0.44 / 0.38, **CPR 0.80 / 0.64 (+0.16)**). 그러나 이어진 S10f는 다시 45,000 episode에 0/4, goal 6-8칸 버킷 51-62%로 그대로였다. 발판을 거친 체크포인트로 S10e를 다시 재 보면 전체 성공률이 0.85 → 0.81로 오히려 떨어졌다. 로그 `logs/comm_scratch_gru_train_part7_S10f1_S10f.log`. |
| 2026-10-21 | **판단: 마지막 obstacle 레벨(S10e, S10, S11)의 통과 기준을 완화.** 남은 약점은 "큰 target + goal 9칸 이상 + obstacle 2개" 조합 하나뿐이고, 시간 제한도 아니고 풀 수 없는 배치도 아니며, 발판 4개(S10f, S10f1, max_steps 변경 2번)로도 오르지 않았다. 이 조합 하나 때문에 커리큘럼 끝(S10, S11)에 못 가는 것보다 기준을 바꾸고 기록하는 편이 낫다고 봤다. 새 기준: **최근 400 episode 전체 성공률 85% 이상 + 모든 버킷 60% 이상** (이전 레벨은 그대로 모든 버킷 85%). 코드: `BucketedCurriculum(overall_threshold=...)`, 레벨 설정 `bucket_threshold` / `overall_threshold` (`comm_scratch/curriculum.py`의 `FINAL_BUCKET_THRESHOLD`, `FINAL_OVERALL_THRESHOLD`). S10e 100k 체크포인트에서 S10e(레벨 23)부터 재시작: `--start-level 23 --start-checkpoint checkpoints\comm_scratch_gru_S10e_100k.pt` (나머지 옵션은 아래와 같음). S10f1/S10f는 레벨 목록에 남지만 이번 실행은 그 뒤에서 시작하므로 거치지 않는다. |

```
.venv\Scripts\python.exe -u -m train.stages.comm_scratch.train --recurrent --start-level 15 ^
  --start-checkpoint checkpoints\comm_scratch_S10a1_clear_obstacle_small_target_scaffold_mastered.pt ^
  --num-envs 384 --num-workers 22 --rollout-steps 64 --minibatch-size 4096 --device cuda ^
  --max-episodes-per-level 200000 > logs\comm_scratch_gru_train.log 2>&1
```

설정 메모: env 384개 x rollout 64 x 에이전트 4 = update당 98,304 transition, minibatch 4,096이면
epoch당 24번, update당 96번 gradient step (target_kl로 조기 종료 가능). 같은 episode 수에서 update
횟수가 MLP 기본 설정(env 16개)보다 훨씬 적어서 정체 판정을 레벨당 200,000 episode로 늘렸다.
MLP 기준선과 batch 설정이 다르다는 점은 비교할 때 감안해야 한다.

## 1. 현재 상태 (2026-10-20)

Track B는 **GRU로 S10a2부터 학습 중**이다 (0장 "자율 진행 기록"). Track A는 8C0s1에서 상한(재개 후
60,000 episode)에 걸려 멈춘 상태 그대로다. 아래 표는 MLP 기준 마지막 상태다.

| | Track A `stage8_comm` | Track B `comm_scratch` |
|---|---|---|
| 출발점 | `stage_8A_size_mastered.pt` (Stage 0-8A 계열, 통신 없이 학습됨) | 무작위 초기화 |
| 통신 | 8B 레벨부터 on | 처음부터 on |
| 마지막 통과 레벨 | **8B4** (크기 + 위치 완전 랜덤, obstacle 없음) | **S10a1** (작은 target으로 경로 막은 obstacle 치우기) |
| 상태 | 8C0s1 (S10a1과 같은 발판)에서 정체로 멈춤 (버킷 57/59/70%). `--resume`하면 8C0s1을 다시 60,000 episode | 멈춤. `--resume`하면 S10a2부터 |
| 통과한 최고 난도 ablation | 8B4: on 0.86 / off 0.61 (**+0.25**) | S9: on 0.89 / off 0.47 (**+0.43**) |

주요 체크포인트 (`checkpoints/`):

| 용도 | Track A | Track B |
|---|---|---|
| 마지막 통과 레벨 | `stage8_comm_stage_8B4_size_and_position_comm_mastered.pt` | `comm_scratch_S10a1_clear_obstacle_small_target_scaffold_mastered.pt` |
| obstacle 없는 최종 모델 | 위와 같음 | `comm_scratch_S9_split_1near_mastered.pt` |
| 일시정지 지점 (2026-09-29) | `stage8_comm_paused_8C0s1.pt` | `comm_scratch_paused_S10a1.pt` |
| 마지막 학습 지점 (2026-10-20) | `stage8_comm_stage_8C0s1_clear_obstacle_small_target_comm_stalled.pt` (= `_latest.pt`) | `comm_scratch_latest.pt` (S10a1 통과 직후) |
| 쓰지 말 것 | `stage8_comm_8C0a_collapsed.pt` (붕괴) | `comm_scratch_S4_collapsed_15k.pt` (붕괴) |

### 실행 방법

```
python -m train.stages.stage_8_comm.train [--resume] [--start-level N --start-checkpoint PATH] [--stop-after-current-level]
python -m train.stages.comm_scratch.train [--resume] [--start-level N --start-checkpoint PATH] [--stop-after-current-level]
```

- `--resume`: `checkpoints/<run>_latest.pt/.json`에서 레벨까지 그대로 이어감.
- 진행 상황: `logs/<run>_train.log`, `logs/<run>_status.json`(현재 레벨, 버킷별 성공률, 통과 이력).
- TensorBoard: `python -m tensorboard.main --logdir runs --port 6006`. 지금 학습 중인 것만 보려면
  실행 필터에 `^(stage8_comm|comm_scratch)$`. 재시작 때 이전 기록은 `runs/<run>_attemptN`,
  `runs/<run>_partN`으로 옮겨 두므로 같은 화면에서 비교할 수 있다.
- 레벨 통과 시 ablation 결과: `evaluation_results/<run>/ablation_<level>_mastered.json`.

---

## 2. 환경에서 알아둘 제약

- **성공 조건**은 `target.cells() == goal_cells` 완전 일치. goal은 항상 target과 같은 모양으로 만든다.
- **밀기 규칙**: 미는 방향 반대쪽 면의 **모든** 칸에 에이전트가 서서 같은 step에 같은 방향을 골라야
  한다 (`env/object.py::contact_cells`). 3x2를 아래로 밀려면 3명, 옆으로 밀려면 2명. 폭 4칸 target을
  세로로 밀려면 4명 전원이 필요하다.
- **벽에 붙은 target은 벽 반대 방향으로 밀 수 없다** (밀 면이 맵 밖). 그래서 모든 생성기는 target을
  벽에서 1칸 이상 떨어뜨린다. goal은 벽에 붙어도 된다.
- **obstacle도 같은 규칙으로 밀린다.** target이 obstacle 쪽으로 밀리면 연쇄로 밀리지 않고 막힌다.
- **관측**: 5x5 로컬 시야 + 자기 좌표 + 직전 step에 받은 HELP 메시지. target/goal의 전역 위치는 없다.
  정책은 **메모리 없는 MLP**라서 시야 밖으로 나간 것은 기억하지 못한다.
- **HELP 메시지**는 "발신자의 나에 대한 상대 좌표"를 거리 제한 없이 전원에게 방송한다. target/goal 좌표를
  직접 보내는 필드는 없다. 자연스러운 용도는 "target(또는 goal)을 본 에이전트가 나머지를 불러모으기"다.
  `help_penalty`(기본 0.002)는 보낸 에이전트에게만 적용된다.

---

## 3. 코드 구성

| 파일 | 역할 |
|---|---|
| `train/stages/comm_runner.py` | 두 트랙 공용 러너. env 16개를 batch로 돌림, 레벨별 mastery, ablation, rehearsal, 재개, 상태 기록 |
| `train/stages/stage_8_comm/` | Track A 레벨 정의(`LATERAL_LEVELS`, `MAIN_LEVELS`)와 실행 스크립트 |
| `train/stages/comm_scratch/` | Track B 생성기(`ScratchScenarioGenerator`), 레벨 정의(`LEVELS`), 실행 스크립트 |
| `train/stages/stage_8/scenario.py` | Stage 8 생성기 (`target_wall_margin=1` 수정 포함) |
| `train/stages/bucketed_curriculum.py` | 버킷별 독립 mastery 판정 |
| `train/ppo.py`, `train/network.py` | 이동/통신 2-head 정책, PPO |
| `train/recurrent_ppo.py`, `train/recurrent_buffer.py` | GRU 메모리 정책용 PPO와 시퀀스 rollout 버퍼 (아래 "GRU 실험 준비") |
| `tests/test_comm_curricula.py` | 두 트랙 레벨·생성기·PPO 변경 테스트 |

### 러너 기능 (`RunConfig`)

| 기능 | 설명 |
|---|---|
| 벡터화 | env 16개 x 에이전트 4명을 forward 한 번으로 처리. 초당 env-step 약 870 → 약 3,800. CPU가 GPU보다 빠르다(병목이 Python env). |
| mastery | 버킷별 최근 100 episode 성공률이 모두 0.85 이상 + 레벨당 최소 1,000 episode |
| 정체 | 레벨당 60,000 episode를 넘기면 `_stalled.pt`를 남기고 멈춤 |
| ablation | 레벨 통과(또는 정체) 때마다 같은 시나리오 300개를 통신 on/off로 비교 |
| `rehearsal_prob=0.25` | episode의 25%를 이미 통과한 레벨에서 뽑아 학습 (mastery 통계에서는 제외) |
| `critic_warmup_updates` | warm-start 직후 처음 N번은 critic head만 학습 (Track A 30) |
| `train_settings()` / `env_settings()` | 레벨별 이동 entropy 계수, `max_steps`, `help_penalty` |
| `stop_after_current_level` | 지금 레벨을 통과하면 멈춤 |
| 저장 | 임시 파일 + `os.replace` + 재시도 (Windows 파일 잠금 대응) |

### PPO 변경

- 통신 2-way head 추가 (`get_action_and_value`, `evaluate_actions`, `act_batch`).
- `comm_entropy_coef`: 이동/통신 head의 entropy 계수 분리.
- `target_kl`: update 중 approx KL이 1.5배를 넘으면 남은 epoch 생략.
- `update(update_policy=False)`: critic head만 학습 (warmup용).
- 두 트랙 공통 설정: `lr=1e-4`, `target_kl=0.02`, 이동 entropy 0.01(발판 레벨만 0.03),
  통신 entropy 0.01, `device="cpu"`.

### GRU 실험 준비 (2026-10-20)

메모리 없는 MLP가 시야 밖으로 나간 target/goal과 직전 step에만 유효한 HELP를 기억하지 못해서,
obstacle 치우기처럼 여러 단계가 필요한 레벨(Track B S10a2, Track A 8C0s1)에서 정체했다는 가설을
시험하기 위한 준비다.

- **`RecurrentActorCritic`** (`train/network.py`): MLP encoder 뒤에 잔차 GRU 분기를 붙인다
  (feature + W·h). W를 0으로 초기화하므로 **MLP 체크포인트를 불러온 직후에는 MLP와 출력이 완전히
  같다.** 파라미터 이름이 ActorCritic과 같아서 MLP 체크포인트가 그대로 로드되고, 새로 생기는 것은
  `rnn`(GRUCell, hidden 128)과 `memory_proj`뿐이다.
- **`RecurrentPPO`** (`train/recurrent_ppo.py`): PPO와 같은 손실(clip, 이동/통신 entropy 분리,
  `target_kl`, critic warmup). rollout 때 매 step의 hidden을 저장하고 길이 32 조각으로 다시 펼치는
  truncated BPTT. minibatch 크기(512 transition)와 update당 gradient step 수는 MLP와 같다.
- **러너**: `RunConfig.recurrent=True`일 때만 새 경로. episode가 끝난 행은 다음 step 직전에 hidden을
  0으로 초기화한다. ablation 평가도 hidden을 들고 다닌다. MLP 경로는 바뀌지 않았다.
- **실행**: 두 train 스크립트에 `--recurrent`. 실행 이름에 `_gru`가 붙어 체크포인트/로그/TensorBoard가
  MLP와 섞이지 않는다.
- **속도**: env 16개 기준 MLP 3,616, GRU 1,688 env-step/s (약 2.1배 느림). 같은 체크포인트에서
  시작한 처음 500 episode 성능은 둘이 같았다 (0-init 확인).
- **테스트**: `tests/test_recurrent.py` (0-init 동등성, step과 unroll 일치, 저장한 hidden에서 다시
  펼친 log_prob 일치, GAE 종료 처리, KL 조기 종료, critic warmup, MLP 체크포인트 로드, 러너
  정지/재개를 MLP와 GRU 모두).

실험 계획 (같은 체크포인트에서 MLP와 GRU 비교):

| 트랙 | 명령 | 비교 대상 (MLP) |
|---|---|---|
| B | `python -m train.stages.comm_scratch.train --recurrent --start-level 15 --start-checkpoint checkpoints/comm_scratch_S10a1_clear_obstacle_small_target_scaffold_mastered.pt` | 같은 체크포인트에서 시작한 MLP S10a2: 53,800 episode 동안 큰 target 0-4%, 작은 target 29-37% (`logs/comm_scratch_train.log`) |
| A | `python -m train.stages.stage_8_comm.train --recurrent --start-level 16 --start-checkpoint checkpoints/stage8_comm_stage_8C0s1_clear_obstacle_small_target_comm_stalled.pt` | MLP 8C0s1: 약 85,000 episode에 버킷 57-70%로 정체 |

### 협동 지표 (RC, CPR)

`evaluation/coordination_metrics.py`. 레벨 통과/정체 시 ablation과 `evaluation/comm_ablation_report.py`가
통신 on/off 각각에 대해 성공률과 함께 기록한다.

- **유효 방향 D_t**: target을 goal 쪽으로 가깝게 만드는 방향 (target과 goal 좌상단의 x, y 차이 부호.
  같은 행/열이면 한 방향, 대각선이면 두 방향). "엉뚱한 면을 채운 것"이 점수가 되지 않게 한다.
- **RC** (모임): 매 step `C_t(o,d) = |F_t(o,d) ∩ P_t| / |F_t(o,d)|`를 유효 방향마다 계산해 최댓값을
  취하고, 유효 방향이 있는 모든 step에 대해 평균한다. F_t(o,d)는 target o를 d로 밀 때 서야 하는
  접촉 칸, P_t는 에이전트가 있는 칸.
- **CPR** (동기화된 밀기): 분모는 어떤 유효 방향 d의 접촉 칸이 모두 차 있고 target이 d로 움직일
  공간(맵 안, 벽/다른 물체 없음)도 있는 step, 분자는 그중 접촉 칸의 에이전트 전원이 d를 고른 step.
  행동 선택 기준이라, 다른 에이전트와 칸이 겹쳐 실제 이동이 취소된 경우도 동기화로 센다.
- 평가 episode의 모든 step을 합쳐 계산한다 (episode 평균이 아니라 step 가중).

### 맵 생성 방식

학습 맵은 **매 episode 새로 무작위 생성**한다 (고정 학습 맵 목록 없음). mastery 판정도 새로 생성된
episode로 한다. ablation은 평가마다 고정 seed로 시나리오 300개를 만들어 통신 on/off에 같은 목록을 쓴다.
학습에 쓰지 않은 고정 테스트 세트(held-out benchmark)는 아직 없다.

---

## 4. Track A — `stage8_comm`

`stage_8A_size_mastered.pt`(크기 랜덤, 위치 고정, 통신 없음)에서 이어서 통신을 켠다. Stage 8의
기하학을 그대로 쓰되, Stage 0-7에서 배운 적 없는 동작을 먼저 가르치는 레벨(8B0*)을 앞에 붙였다.
`episode`는 그 레벨을 마지막으로 통과할 때 걸린 수다.

| # | 레벨 | 내용 | episode | 통신 on | off |
|---|---|---|---|---|---|
| 0 | 8B0sL | 옆으로 밀기 발판, 왼쪽 75%, 2명을 옆면에 미리 세움, max_steps 60 | 3,670 | 0.93 | 0.94 |
| 1 | 8B0s | 옆으로 밀기 발판, 좌우 균등 | 1,000 | 0.98 | 0.96 |
| 2 | 8B0 | 옆으로만 밀기, 미리 세우지 않음 | 1,000 | 0.98 | 0.93 |
| 3 | 8B0a | goal 아래 7칸 + 옆 1칸 | 1,064 | 0.97 | 0.93 |
| 4 | 8B0b | goal 아래 7칸 + 옆 2칸 | 1,000 | 0.93 | 0.91 |
| 5 | 8B0uS2 | 위로 밀기 발판, 2칸 폭 target, 2명 미리 세움 | 1,629 | 0.92 | 0.83 |
| 6 | 8B0uS | 위로 밀기 발판, 3칸 폭 | 1,772 | 0.92 | 0.80 |
| 7 | 8B0u | 위로 밀기, 미리 세우지 않음 | 1,000 | 0.98 | 0.95 |
| 8 | 8B0uLs | 위로 1-3칸, target 위치 랜덤, 발판 | 2,607 | 0.96 | 0.87 |
| 9 | 8B0uL | 위로 2-5칸, target 위치 랜덤 | 1,000 | 0.97 | 0.95 |
| 10 | 8B0tS | 세로로 긴 target(2x3/3x3/4x3) 옆으로, 3명 발판 | 3,723 | 0.95 | 0.87 |
| 11 | 8B0t | 같은 배치, 미리 세우지 않음 | 1,000 | 0.94 | 0.96 |
| 12 | 8B1 | target 위치 반경 2 | 1,000 | 1.00 | 0.96 |
| 13 | 8B2 | 반경 5 | 1,153 | 0.89 | 0.87 |
| 14 | 8B3 | 위치 완전 랜덤 | 3,857 | 0.83 | 0.67 |
| 15 | **8B4** | 크기 + 위치 완전 랜덤 | 40,611 | 0.86 | 0.61 |
| 16 | 8C0s1 | 경로 막은 1x1 obstacle 치우기, 작은 target(2x1, 2x2), 근처 스폰 | **정체** (약 85,000, 버킷 57-70%) | 0.63 | 0.60 |
| 17 | 8C0s | 같은 발판, 모든 크기 | | | |
| 18 | 8C0a | goal 2-5칸 + 작은 obstacle 1개 | (이전 정체: 0.76 / 0.66) | | |
| 19 | 8C0a2 | goal 무제한 + 작은 obstacle 1개 | | | |
| 20 | 8C0b | 보통 크기 obstacle 1개 | | | |
| 21 | 8C | obstacle 0-3개 | | | |
| 22 | 8D | obstacle 0-4개 | | | |
| 23-25 | 9 split | 8D 위에 split spawn 3near → 2near → 1near | | | |

발판 레벨(0, 1, 5, 6, 8, 10, 16, 17)과 가로/위로 밀기 레벨(0-11)은 이동 entropy 0.03, 나머지는 0.01.
가로/위로 밀기 전용 레벨에는 8A 과제(goal 바로 아래)를 30-50% 섞는다 (`below_mix`, 망각 방지).

---

## 5. Track B — `comm_scratch`

무작위 초기화에서 새 커리큘럼으로 학습한다. 설계 원칙:

1. target 위치는 첫 레벨부터 맵 전체에서 랜덤. 난이도는 크기 / goal 거리 / 스폰 / obstacle로 조절한다.
2. 한 레벨에서는 한 축만 바꾼다.
3. 통신은 처음부터 켜되, goal이 시야 안에 있는 동안(S0-S3b)은 `help_penalty=0`, S4부터 0.002.
4. split spawn은 위치가 이미 랜덤인 상태 위에 얹는다.

| # | 레벨 | 바뀌는 축 | episode | 통신 on | off |
|---|---|---|---|---|---|
| 0 | S0 | 2x1 target, goal 2-3칸, 근처 스폰(r2) | 31,470 | 0.84 | 0.75 |
| 1 | S1 | goal 2-5칸 | 1,050 | 0.87 | 0.84 |
| 2 | S2 | 3x2 target | 3,673 | 0.84 | 0.67 |
| 3 | S3s | 발판: 세로로 긴 target 옆으로 1-2칸, 3명 미리 세움 | 20,239 | 0.96 | 0.97 |
| 4 | S3t | 같은 배치, 근처 랜덤 스폰 | 6,319 | 0.99 | 1.00 |
| 5 | S3a | 크기 2-3 x 2-3 | 5,839 | 0.83 | 0.70 |
| 6 | S3b | 크기 2-4 x 2-3 | 7,835 | 0.81 | 0.62 |
| 7 | S4 | goal 2-8칸, help_penalty 켬 | 19,896 | 0.82 | 0.66 |
| 8 | S5 | goal 무제한 | 47,188 | 0.92 | 0.55 |
| 9 | S6 | 스폰 반경 5 | 4,844 | 0.90 | 0.56 |
| 10 | S7 | 스폰 완전 랜덤 | 4,980 | 0.90 | 0.60 |
| 11 | S8 | split 2near/2far | 7,538 | 0.89 | 0.52 |
| 12 | **S9** | split 1near/3far | 5,029 | 0.89 | 0.47 |
| 13 | **S10a** | 근처 스폰, goal 2-5칸, 작은 obstacle 1개 | 약 67,000 | 0.83 | 0.45 |
| 14 | **S10a1** | 발판: 경로 막은 1x1 obstacle, 작은 target(2x1, 2x2) | 약 29,900 | 0.78 | 0.70 |
| 15 | S10a2 | 같은 발판, 모든 크기 | | | |
| 16 | S10b | goal 무제한 (근처 스폰, 작은 obstacle) | | | |
| 17 | S10c | 스폰 완전 랜덤 | | | |
| 18 | S10d | obstacle 보통 크기(1-3 x 1-2) | | | |
| 19 | S10 | split 1near + obstacle 0-2개 | | | |
| 20 | S11 | obstacle 0-4개 | | | |

이동 entropy 0.03은 S3s, S3t, S10a, S10a1, S10a2에만 쓴다.

---

## 6. 통신 효과 (ablation)

같은 체크포인트, 같은 시나리오 300개에서 통신을 켠 성공률과 끈(HELP를 NO_MESSAGE로 강제) 성공률의 차이.

| 조건 | 격차 |
|---|---|
| 모두 target 근처에서 시작, goal 가까움 (Track B S1, S3s, S3t; Track A 8B0, 8B1) | 0에서 +0.04 |
| 큰 target, 여러 명을 모아야 함 (Track B S2, S3a, S3b) | +0.12에서 +0.18 |
| 미는 에이전트가 goal을 못 봄 (Track A 위로 밀기 8B0uS2, 8B0uS) | +0.09에서 +0.12 |
| target/goal이 맵 전체 랜덤 (Track A 8B3, 8B4) | +0.16에서 +0.25 |
| goal 무제한 + 스폰 분산 (Track B S5-S8) | +0.30에서 +0.36 |
| **target을 본 에이전트 1명 + 멀리 3명 (Track B S9)** | **+0.43** |
| 작은 obstacle 1개 (Track B S10a) | +0.39 |

**결론.** 두 모델 모두 통신을 실제로 쓰는 협동을 학습했다. 격차는 target/goal이 시야 밖에 있거나
여러 명을 불러모아야 할 때 커지고, 모두가 target 근처에서 시작하면 거의 없다. Track B S9에서 통신을
끄면 성공률이 절반 아래로 떨어진다 — 원래 Phase 2가 목표로 한 "발견자에게 집결" 프로토콜의 가장 강한
증거다. 단, 어떤 메시지 패턴이 학습됐는지는 아직 직접 분석하지 않았다.

### MLP 체크포인트의 RC / CPR / 모이는 속도 (2026-10-20)

`evaluation/comm_ablation_report.py`로 통과 체크포인트를 그 레벨에서 다시 평가했다 (300 episode,
통신 on / off). "모이기 step"은 처음으로 유효 push가 가능한 접촉 배치가 될 때까지의 step 수
(끝까지 없으면 episode 길이). 원본: `evaluation_results/<run>/comm_report.md`.

| 레벨 | 성공률 on / off | RC on / off | CPR on / off | 모이기 step on / off |
|---|---|---|---|---|
| B S0 | 0.83 / 0.77 | 0.40 / 0.35 | 0.23 / 0.24 | 5.2 / 8.1 |
| B S2 | 0.82 / 0.67 | 0.47 / 0.47 | 0.32 / 0.24 | 26.5 / 34.5 |
| B S3b | 0.81 / 0.62 | 0.49 / 0.48 | 0.42 / 0.38 | 32.0 / 55.0 |
| B S5 | 0.92 / 0.55 | 0.53 / 0.47 | 0.56 / 0.52 | 13.8 / 29.5 |
| B S7 | 0.90 / 0.60 | 0.47 / 0.44 | 0.61 / 0.61 | 22.6 / 51.5 |
| B S8 | 0.89 / 0.52 | 0.49 / 0.46 | 0.54 / 0.59 | 18.6 / 55.6 |
| **B S9** | 0.89 / 0.47 | 0.48 / 0.44 | 0.58 / 0.57 | **22.7 / 68.5** |
| B S10a | 0.83 / 0.45 | 0.55 / 0.42 | 0.36 / 0.31 | 29.3 / 74.1 |
| B S10a1 | 0.78 / 0.70 | 0.57 / 0.52 | 0.50 / 0.47 | 15.1 / 19.5 |
| A 8B1 | 1.00 / 0.96 | 0.54 / 0.59 | 0.78 / 0.65 | 15.3 / 16.4 |
| A 8B3 | 0.83 / 0.67 | 0.42 / 0.45 | 0.52 / 0.50 | 23.9 / 40.6 |
| A 8B4 | 0.86 / 0.61 | 0.39 / 0.39 | 0.60 / 0.63 | 27.5 / 43.5 |

**해석.** 통신이 바꾸는 것은 "모인 뒤 동시에 미는 능력"(CPR, 거의 같음)이나 "밀 자리에 서 있는 시간
비율"(RC, 조금 높음)이 아니라 **필요한 인원을 모으는 속도**다. 스폰이 흩어질수록 차이가 커져서, target을
본 에이전트가 1명뿐인 S9에서는 통신을 켜면 3배 빨리 모인다 (22.7 / 68.5 step). 모두 target 근처에서
시작하는 레벨(S3s, S3t)은 차이가 없다. RC는 step 평균이라 찾아다니는 시간이 길어도 비율로는 크게
드러나지 않는다 — 그래서 모이는 속도 지표를 따로 둔다.

### GRU 체크포인트 (Track B `comm_scratch_gru`)

| 레벨 | 성공률 on / off | RC on / off | CPR on / off | 모이기 step on / off |
|---|---|---|---|---|
| S10a2 | 0.83 / 0.60 | 0.57 / 0.46 | 0.61 / 0.51 | 31.1 / 45.3 |
| S10b | 0.86 / 0.61 | 0.56 / 0.48 | 0.67 / 0.65 | 15.0 / 27.4 |
| S10c | 0.88 / 0.58 | 0.50 / 0.45 | 0.65 / 0.59 | 24.7 / 51.5 |

원본: `evaluation_results/comm_scratch_gru/comm_report.md`. GRU에서도 통신의 주효과는 모이는 속도
(S10c 24.7 / 51.5 step)이고, MLP보다 RC와 CPR의 통신 차이가 조금 더 크다 (CPR +0.02에서 +0.10).

---

## 7. 교훈

1. **풀 수 없는 episode부터 의심한다.** Stage 8 생성기가 target을 벽에 붙여 놓아, 8B1의 19%, 8B3 이후
   32-34%가 원리적으로 풀 수 없었다. 버킷 기준 0.85를 넘을 수 없었던 Stage 8 8B1 실패의 한 원인.
2. **"위치 랜덤화"는 숨은 새 기술을 요구한다.** Stage 0-7은 goal이 항상 target 바로 아래였고 target이
   (4,2)였다. 그 결과 옆으로 밀기 0%, 위로 밀기 0%였고, 위로 밀기를 가르쳐도 (4,2)에서만 동작했다.
   방향마다, 그리고 랜덤 위치에서 따로 가르쳐야 했다.
3. **한 레벨에 한 축.** 여러 축을 동시에 바꾼 레벨(Stage 8 8B1 2차, Track B S3, S10b 원안, Track A
   8C0a 원안)은 모두 막혔다. 특히 새 축은 쉬운 기반(goal 가까움, 근처 스폰) 위에서 들여야 한다.
4. **여러 명이 동시에 미는 새 동작에는 발판이 필요하다.** 밀어야 할 면에 에이전트를 미리 세우고
   episode를 짧게(60-100 step) 끊는 발판 → 미리 세우지 않는 단계 → 원래 레벨 순서로 풀렸다
   (가로 밀기, 위로 밀기, 세로로 긴 target 옆으로 밀기). 작은 target(2칸 폭)부터 시작하면 더 쉽다.
   이 발견 단계에서만 이동 entropy 0.03이 필요했다.
5. **망각.** 새 레벨이 이전 기술을 쓰지 않으면 통째로 잊는다 (가로 전용 레벨 후 아래로 밀기 0%).
   이전 과제를 섞고(`below_mix`, rehearsal 25%) 그 버킷을 mastery 조건에 넣는다.
6. **학습 안정성.** lr 3e-4는 후반 레벨에서 이미 하던 과제까지 잃는 붕괴를 반복했다 → lr 1e-4 +
   `target_kl=0.02`. entropy 0.03은 짧은 발판 레벨에서만 안전하고, 긴 레벨에서는 붕괴를 부른다.
   warm-start 직후에는 critic이 이전 성공률에 맞춰져 있어서 critic warmup이 필요했다.
7. **실패를 쪼개서 본다.** 정체 원인은 모두 "버킷 평균"이 아니라 방향, 모양, obstacle 위치별로 나눠
   봤을 때 드러났다 (예: 8B2 정체 = goal이 target 위에 있는 경우만 0/58).

---

## 8. 시행착오 기록

### Track A

| 차수 | 문제 | 진단 | 조치 | 로그 / 체크포인트 |
|---|---|---|---|---|
| 1 | 8B1 24% → 5% | 통신 on/off 둘 다 하락 → fine-tune 자체가 망가뜨림. critic이 "거의 항상 성공"에 맞춰짐 | critic warmup 30회, lr 1e-4 | `logs/stage8_comm_train_attempt1_degraded.log` |
| 2 | 8B1 20% 정체 | goal이 바로 아래면 88%, 옆으로 1칸만 어긋나도 0/230 | 옆으로 밀기 레벨 8B0a/8B0b 추가 | `..._attempt2_no_lateral.log` |
| 3 | 8B0a에서 옆 0% | 아래로 밀기만으로 보상 대부분을 받음 | 같은 행 옆으로만 미는 8B0 추가 | `..._attempt3_8B0a_no_lateral.log` |
| 4 | 8B0 0% | 8A 정책은 어떤 배치에서도 아래로만 밂 (3,000번 중 옆 0번) | 2명을 옆면에 세우는 발판 + entropy 0.03 (0.01은 0%) | `..._attempt4_8B0_no_lateral.log` |
| 5 | 8B0a에서 dx=0까지 0% | 아래로 밀기를 잊음 | 가로 레벨에 8A 과제 50% 혼합 | `checkpoints/archive_stage8_comm_attempt5_forgetting/` |
| 6 | 왼쪽 밀기 0% | 오른쪽 면 2명의 LEFT 확률 3-4%, 대신 RIGHT 52% | 왼쪽 75% 발판 8B0sL, max_steps 60 | `..._attempt6_no_left.log` |
| 7 | 8B2 정체 (먼 버킷 50%) | goal이 target 위면 0/58, 위로 민 적 0번 | 위로 밀기 발판 추가 | `..._attempt7_8B0sL_to_8B2.log` |
| 8 | 3칸 폭 위로 밀기 발판 0% | 미는 에이전트가 goal을 못 봄 + 3명 동시 UP 확률 0.08% | 2칸 폭 발판 8B0uS2 먼저 | `..._attempt8_uS_3wide.log` |
| 9 | 8B3 정체 (위로 0/63) | 위로 밀기가 (4,2) 좌표에 묶임 | target 위치 랜덤인 위로 밀기 8B0uLs/8B0uL | `..._attempt9_uS2_to_8B3.log` |
| 10 | 8B4에서 세로 3칸 target 11-21% | 3명이 옆면에 줄 서서 밀기 | 세로로 긴 target 발판 8B0tS/8B0t → 8B4 통과 | `..._attempt10_uLs_to_8B4.log` |
| 11 | 8C obstacle 3개 0-3% | obstacle을 한 번에 0-3개 | 8C 앞에 obstacle 1개 레벨 | `..._attempt11_tS_to_8C.log` |
| 12 | 8C0a(entropy 0.03) 붕괴 | 긴 레벨에서 0.03 | entropy 0.01 + `target_kl` | `..._attempt12_8C0a_collapse.log` |
| 13 | 8C0a 23-39% 정체 | 가장 어려운 기하학 위에 obstacle | goal 2-5칸으로 쉬운 기반 | `..._attempt13_8C0a_hard.log` |
| 14 | 8C0a 상한 도달 (73-83%) | obstacle이 경로 위면 1/15 | obstacle 치우기 발판 8C0s | `..._attempt14_8C0a_stalled.log` |
| 15 | 8C0s 26-31% | 큰 target은 통로가 3칸 높이 | 작은 target 발판 8C0s1 | `..._attempt15_8C0s.log` |
| 15b | 8C0s1 정체 (버킷 57-70%, 상한 도달) | 같은 발판을 Track B는 약 29,900 episode에 통과. 원인 미분석 | 아직 없음 | `..._attempt15b_8C0s1_part1.log`, `logs/stage8_comm_train.log` |

로그 파일은 모두 `logs/stage8_comm_train_*`.

### Track B

| 부 | 문제 | 진단 | 조치 | 로그 / 체크포인트 |
|---|---|---|---|---|
| 1 | S0이 31,000 episode | 가로(1명) 96%, 세로(2명 동시) 42% | 시간이 해결 | `logs/comm_scratch_train_part1_S0-S3.log` |
| 1 | S3 붕괴 (작은 target까지 25-39%) | 크기 2-4 x 2-3을 한 번에 (4명 전원 필요) | S3a/S3b 분할 + rehearsal | 같은 로그 |
| 2 | 저장 중 프로세스 사망 | Windows 오류 1224 (파일 잠금) | `_safe_save` | `..._part2_crash.log` |
| 4-6 | S3a 정체 | 세로로 긴 target 옆으로 밀기 22% | 발판 S3s → 중간 S3t → S3a 통과 | `..._part4_*`, `..._part5_*`, `..._part6_S3s.log` |
| 7 | S4 붕괴 (S3b 과제 85% → 18%) | help_penalty 아님 (성능 하락이 먼저). lr 3e-4 | lr 1e-4 + `target_kl` → S4-S9 무붕괴 | `..._part7_S3t_to_S4collapse.log`, `checkpoints/comm_scratch_S4_collapsed_15k.pt` |
| 8 | S10 obstacle 7-17% | obstacle 0개 88%, 1개 34%, 2개 9% | S10a(쉬운 기반 + obstacle 1개) | `..._part8_S4_to_S10.log` |
| 10 | S10b 18-41% | 세 축을 한꺼번에 바꿈 | S10b/S10c/S10d로 분할 | `..._part10_S10a_S10b3axes.log` |
| 11 | 새 S10b 42-64% 정체 | 경로 위 obstacle 0-38%, 경로 밖이어도 먼 goal 45-63% | obstacle 치우기 발판 S10a2 | `..._part11_S10b.log`, `checkpoints/comm_scratch_S10b_45k.pt` |
| 12 | S10a2 큰 target 0-4% | 통로가 3칸 높이 | 작은 target 발판 S10a1 → 약 29,900 episode에 통과 (2026-10-20) | `..._part12_S10a2.log`, `..._part13_S10a1_part1.log`, `logs/comm_scratch_train.log` |

로그 파일은 모두 `logs/comm_scratch_train_*`.

---

## 9. 다음에 할 일 / 열린 질문

- **obstacle 단계 마무리.** Track B는 작은 target 발판(S10a1)을 통과했고 다음은 모든 크기 발판(S10a2).
  Track A는 작은 target 발판(8C0s1)에서 정체했다 (약 85,000 episode, 버킷 57-70%). 같은 발판을
  Track B는 통과했으므로, Track A 쪽 실패 유형부터 나눠 볼 필요가 있다. 큰 target까지 넘어가면
  남은 obstacle 레벨(Track A 8C0a-8D, Track B S10b-S11)과 Track A의 split 레벨이 남는다.
- **no-comm 대조군.** 같은 커리큘럼을 `comm_enabled=False`로 학습해서, "학습 후 통신을 끈 모델"이
  아니라 "통신 없이 학습한 모델"과 비교한다.
- **고정 테스트 세트.** 학습에 쓰지 않은 고정 시나리오 세트로 두 트랙과 대조군을 함께 평가한다
  (`evaluation/benchmark_scenarios.py` 확장).
- **HELP 사용 패턴 분석.** 누가, 언제(target/goal을 본 상태인지), 받은 쪽이 어떻게 움직이는지.
- **통신을 mastery 조건에 넣을지.** 지금은 ablation을 기록만 한다. `max_steps`를 줄여 통신이 꼭
  필요하게 만드는 튜닝은 하지 않았다.
- **메모리 없는 정책의 한계.** target/goal이 시야 밖으로 나가면 기억하지 못한다. 먼 goal + obstacle
  조합이 가장 약한 것도 이와 관련 있을 수 있다. GRU 실험 준비는 끝났다 (3장 "GRU 실험 준비").
  centralized critic(MAPPO)은 GRU 결과를 본 뒤 하나씩 얹는다 (둘을 동시에 넣으면 무엇이 효과를
  냈는지 구분할 수 없다).
- **Track B 중단 지점.** 세션 종료로 MLP S10a2 실행이 53,800 episode에서 멈췄다 (큰 target 0-4%).
  `comm_scratch_latest.pt`가 그 지점이다.

---

## 부록: 배경 (2026-09-17 - 09-29 이전)

- **Stage 0-6** (`train/stages/stage_0_6`): 고정 좌표 target, obstacle 0-2개, 스폰 반경만 단계적 확대.
- **Stage 7** (`train/stages/stage_7`): Stage 6 기하학을 x축으로 평행 이동 (target과 goal을 같이 이동).
- **Stage 8 (통신 없음)** (`train/stages/stage_8`): 8A(크기 랜덤)는 통과(`stage_8A_size_mastered.pt`).
  8B(위치 랜덤)는 세 번 실패했다 — 완전 랜덤 위치를 한 번에(탐색 실패), 크기와 위치를 동시에(정체),
  크기 고정 반경 2(성공률 40% → 4%). 나중에 원인은 풀 수 없는 episode 19%, entropy 0.03 폭주,
  옆으로 밀기를 배운 적 없음 세 가지로 밝혀졌다.
- **통신 아키텍처 (2026-09-22)**: 정책에 통신 head 추가, `PPO.load(strict=False)`로 comm head 없는
  구 체크포인트를 warm-start, `help_penalty`, `PolicyEvaluator(disable_communication=...)`. 이전의
  모든 학습 스크립트는 env에 `NO_MESSAGE`를 보내 기존 동작을 유지한다. 이때 `PPO.__init__`이
  `ActorCritic`을 위치 인자로 호출해 comm head가 256-way로 만들어지던 버그를 테스트가 잡아냈다.
- **원래 Stage 9 (파이프라인 검증판)** (`train/stages/stage_9`): target 위치를 고정한 채 split spawn만
  얹어 파이프라인이 돌아가는지만 확인했다 (성공률 95-100%, help_rate 20-24%). 위치가 고정이면 통신이
  필요 없으므로 성능 결론은 내리지 않았다. 진짜 필요성 검증은 위치 일반화 이후로 미뤘고, 이것이
  지금의 두 트랙이 됐다.
