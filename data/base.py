import os
import logging
import csv
from torch.utils.data import DataLoader

from .mm_pre import MMDataset
from .text_pre import TextDataset
from .video_pre import VideoDataset
from .audio_pre import AudioDataset
from .mm_pre import MMDataset
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args, logger_name = 'Multimodal Intent Recognition'):
        # 로거 설정
        self.logger = logging.getLogger(logger_name)

        # 벤치마크 데이터 설정
        self.benchmarks = benchmarks[args.dataset]

        # 데이터 경로 설정
        self.data_path = os.path.join(args.data_path, args.dataset)

        # 데이터 모드에 따른 라벨 리스트 설정
        if args.data_mode == 'multi-class':
            self.label_list = self.benchmarks["intent_labels"]
        elif args.data_mode == 'binary-class': 
            self.label_list = self.benchmarks['binary_intent_labels']
        else:
            raise ValueError('The input data mode is not supported.')
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))

        # 라벨 개수 및 각 모달의 특성 및 시퀀스 길이 설정
        args.num_labels = len(self.label_list)        
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = \
            self.benchmarks['feat_dims']['text'], self.benchmarks['feat_dims']['video'], self.benchmarks['feat_dims']['audio']
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = \
            self.benchmarks['max_seq_lengths']['text'], self.benchmarks['max_seq_lengths']['video'], self.benchmarks['max_seq_lengths']['audio']

        # 훈련, 검증, 테스트 데이터 인덱스 및 라벨 아이디 가져오기
        self.train_data_index, self.train_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'train.tsv'), args.data_mode)
        self.dev_data_index, self.dev_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'dev.tsv'), args.data_mode)
        self.test_data_index, self.test_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'test.tsv'), args.data_mode)

        # 단일 모달 특성 가져오기
        self.unimodal_feats = self._get_unimodal_feats(args, self._get_attrs())
        # 다중 모달 데이터 가져오기
        self.mm_data = self._get_multimodal_data(args)
        # 데이터 로더 설정
        self.mm_dataloader = self._get_dataloader(args, self.mm_data)

    def _get_indexes_annotations(self, read_file_path, data_mode):
        # 라벨 매핑 생성
        label_map = {}
        for i, label in enumerate(self.label_list):
            label_map[label] = i

        # 파일 읽기 및 인덱스 및 라벨 아이디 추출
        with open(read_file_path, 'r') as f:
            data = csv.reader(f, delimiter="\t")
            indexes = []
            label_ids = []

            for i, line in enumerate(data):
                if i == 0:
                    continue
                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)
                
                if data_mode == 'multi-class':
                    label_id = label_map[line[4]]
                else:
                    label_id = label_map[self.benchmarks['binary_maps'][line[4]]]
                
                label_ids.append(label_id)

        return indexes, label_ids
    
    def _get_unimodal_feats(self, args, attrs):
        # 각 모달 별 특성 데이터셋 가져오기
        text_feats = TextDataset(args, attrs).feats
        video_feats = VideoDataset(args, attrs).feats
        audio_feats = AudioDataset(args, attrs).feats

        return {
            'text': text_feats,
            'video': video_feats,
            'audio': audio_feats
        }
    
    def _get_multimodal_data(self, args):
        # 다중 모달 데이터 생성
        text_data = self.unimodal_feats['text']
        video_data = self.unimodal_feats['video']
        audio_data = self.unimodal_feats['audio']
        
        mm_train_data = MMDataset(self.train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
        mm_dev_data = MMDataset(self.dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'])
        mm_test_data = MMDataset(self.test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

        return {
            'train': mm_train_data,
            'dev': mm_dev_data,
            'test': mm_test_data
        }

    def _get_dataloader(self, args, data):
        # 데이터 로더 생성
        self.logger.info('Generate Dataloader Begin...')

        train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        
        self.logger.info('Generate Dataloader Finished...')

        return {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }
        
    def _get_attrs(self):
        # 클래스 속성들을 딕셔너리로 추출
        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs
