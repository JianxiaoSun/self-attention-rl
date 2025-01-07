#!/bin/bash -x

# ===========================================================================================
# launch all jobs by passing different combinations of <game, seed, self_attn> to job.sh
# ===========================================================================================

# declare the atari game list
gym_atari_game_list=(
	AdventureNoFrameskip-v4 
	AirRaidNoFrameskip-v4 
	AlienNoFrameskip-v4 
	AmidarNoFrameskip-v4 
	AssaultNoFrameskip-v4 
	AsterixNoFrameskip-v4 
	AsteroidsNoFrameskip-v4 
	AtlantisNoFrameskip-v4 
	BankHeistNoFrameskip-v4 
	BattleZoneNoFrameskip-v4 
	BeamRiderNoFrameskip-v4 
	BerzerkNoFrameskip-v4 
	BowlingNoFrameskip-v4 
	BoxingNoFrameskip-v4 
	BreakoutNoFrameskip-v4 
	CarnivalNoFrameskip-v4 
	CentipedeNoFrameskip-v4 
	ChopperCommandNoFrameskip-v4 
	CrazyClimberNoFrameskip-v4 
	DefenderNoFrameskip-v4 
	DemonAttackNoFrameskip-v4 
	DoubleDunkNoFrameskip-v4 
	ElevatorActionNoFrameskip-v4 
	EnduroNoFrameskip-v4 
	FishingDerbyNoFrameskip-v4 
	FreewayNoFrameskip-v4 
	FrostbiteNoFrameskip-v4 
	GopherNoFrameskip-v4 
	GravitarNoFrameskip-v4 
	HeroNoFrameskip-v4 
	IceHockeyNoFrameskip-v4 
	JamesbondNoFrameskip-v4 
	JourneyEscapeNoFrameskip-v4 
	KangarooNoFrameskip-v4 
	KrullNoFrameskip-v4 
	KungFuMasterNoFrameskip-v4 
	MontezumaRevengeNoFrameskip-v4 
	MsPacmanNoFrameskip-v4 
	NameThisGameNoFrameskip-v4 
	PhoenixNoFrameskip-v4 
	PitfallNoFrameskip-v4 
	PongNoFrameskip-v4 
	PooyanNoFrameskip-v4 
	PrivateEyeNoFrameskip-v4 
	QbertNoFrameskip-v4 
	RiverraidNoFrameskip-v4 
	RoadRunnerNoFrameskip-v4 
	RobotankNoFrameskip-v4 
	SeaquestNoFrameskip-v4 
	SkiingNoFrameskip-v4 
	SolarisNoFrameskip-v4 
	SpaceInvadersNoFrameskip-v4 
	StarGunnerNoFrameskip-v4 
	TennisNoFrameskip-v4 
	TimePilotNoFrameskip-v4 
	TutankhamNoFrameskip-v4 
	UpNDownNoFrameskip-v4 
	VentureNoFrameskip-v4 
	VideoPinballNoFrameskip-v4 
	WizardOfWorNoFrameskip-v4 
	YarsRevengeNoFrameskip-v4 
	ZaxxonNoFrameskip-v4
	)

# declare the list of self-attention types ("NA" means no-self-attention)
self_attn_list=("NA" "SWA" "CWRA" "CWCA" "CWRCA")

# declare the seed list
seed_list=(0 1 10 42 1234)

# launch a job for each <game, self_attn, seed> in the 'research' queue, you may use other queues in your cluster
for game in ${gym_atari_game_list[@]}
do
	for self_attn in ${self_attn_list[@]}
	do
		for seed in ${seed_list[@]}
		do
			qsub -q research -v game=${game},self_attn=${self_attn},seed=${seed} -l select=1:ncpus=16:ngpus=1 job.sh
			sleep 30
		done
	done
done