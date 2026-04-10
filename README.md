# cloud_vmp_optimization

이 저장소는 Virtual Machine Placement(VMP) 문제를 실험하기 위한 연구용 repository입니다. Azure trace 전처리 코드, 여러 toy/prototype 최적화 모델, 그리고 실험별 결과 묶음을 함께 관리합니다.

실험 코드는 `experiments/` 아래에 있고, 가공된 입력 데이터는 `data/processed/` 아래에 정리되어 있습니다. 각 실험은 자신만의 `results/` 폴더를 내부에 두어 결과 그림과 요약 파일을 함께 보관합니다.

가장 최근에 정리된 chance-constrained 2SP toy experiment 문서는 `experiments/2604-chance-2sp-toy/` 아래에 있습니다.
