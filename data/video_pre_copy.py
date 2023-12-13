import os
import numpy as np
import pickle
import logging

__all__ = ['VideoDataset']

class VideoDataset:

    def __init__(self, args, base_attrs):
        # 로거 설정
        self.logger = logging.getLogger(args.logger_name)
        # 비디오 특징 파일 경로 설정
        video_feats_path = os.path.join(base_attrs['data_path'], args.video_data_path, args.video_feats_path)
        # 비디오 특징 파일 존재 여부 확인
        if not os.path.exists(video_feats_path):
            raise Exception('Error: The directory of video features is empty.')
        
        # 비디오 특징 로드
        self.feats = self.__load_feats(video_feats_path, base_attrs)
        # 비디오 특징 패딩 처리
        self.feats = self.__padding_feats(args, base_attrs)
    
    def __load_feats(self, video_feats_path, base_attrs):
        # 비디오 특징 로드 함수
        self.logger.info('Load Video Features Begin...')

        # 비디오 특징 파일 열기 및 로드
        with open(video_feats_path, 'rb') as f:
            video_feats = pickle.load(f)
        
        # 훈련, 검증, 테스트 세트의 비디오 특징 추출
        train_feats = [video_feats[x] for x in base_attrs['train_data_index']]
        dev_feats = [video_feats[x] for x in base_attrs['dev_data_index']]
        test_feats = [video_feats[x] for x in base_attrs['test_data_index']]

        self.logger.info('Load Video Features Finished...')

        # 비디오 특징 반환
        return {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats
        }

    def __padding(self, feat, video_max_length, padding_mode='zero', padding_loc='end'):
        # 비디오 특징 패딩 함수
        # 예시 매개변수: feat=np.array([...]), video_max_length=128, padding_mode='zero', padding_loc='end'
        assert padding_mode in ['zero', 'normal']
        assert padding_loc in ['start', 'end']

        video_length = feat.shape[0]
        # 이미 최대 길이 이상인 경우 자르기
        if video_length >= video_max_length:
            return feat[video_max_length, :]

        # 패딩 모드에 따라 패딩 값 설정
        if padding_mode == 'zero':
            pad = np.zeros([video_max_length - video_length, feat.shape[-1]])
        elif padding_mode == 'normal':
            mean, std = feat.mean(), feat.std()
            pad = np.random.normal(mean, std, (video_max_length - video_length, feat.shape[1]))
        
        # 패딩 위치에 따라 패딩 적용
        if padding_loc == 'start':
            feat = np.concatenate((pad, feat), axis=0)
        else:
            feat = np.concatenate((feat, pad), axis=0)

        return feat

    def __padding_feats(self, args, base_attrs):
        # 비디오 특징 전체에 대한 패딩 처리 함수
        video_max_length = base_attrs['benchmarks']['max_seq_lengths']['video']

        padding_feats = {}

        for dataset_type in self.feats.keys():
            feats = self.feats[dataset_type]

            tmp_list = []

            for feat in feats:
                feat = np.array(feat).squeeze(1)
                padding_feat = self.__padding(feat, video_max_length, padding_mode=args.padding_mode, padding_loc=args.padding_loc)
                tmp_list.append(padding_feat)

            padding_feats[dataset_type] = tmp_list

        return padding_feats
