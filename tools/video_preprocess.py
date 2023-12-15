import os
import torch
import numpy as np
import json
import pickle
import argparse

from torch import nn
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

def parse_arguments():
    # 명령줄 인수를 파싱하는 함수
    parser = argparse.ArgumentParser()
    # 각종 경로와 설정에 대한 인수 추가
    parser.add_argument('--detection_checkpoint_path', type=str, default='mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help="The directory of the detection checkpoint path.")
    parser.add_argument('--detection_config_path', type=str, default='mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help="The directory of the detection configuration path.")
    parser.add_argument('--video_data_path', type=str, default='MIA/datasets/video_data', help="The directory of the video data path.")
    parser.add_argument('--video_feats_path', type=str, default='video_feats_test.pkl', help="The directory of the video features path.")
    parser.add_argument('--frames_path', type=str, default='MIA/datasets/human_annotations/screenshots', help="The directory of human-annotated frames with bbox.")
    parser.add_argument('--speaker_annotation_path', type=str, default='MIA/datasets/human_annotations/speaker_annotations.json', help="The original file of annotated speaker ids.")
    parser.add_argument('--TalkNet_speaker_path', type=str, default='MIA/datasets/speaker_annotation/Talknet', help="The output directory of TalkNet model.")
    parser.add_argument("--use_TalkNet", action="store_true", help="whether using the annotations from TalkNet to get video features.")
    parser.add_argument("--roi_feat_size", type = int, default=7, help="The size of Faster R-CNN region of interest.")

    args = parser.parse_args()

    return args

class VideoFeature:
    
    def __init__(self, args):
        # 비디오 특징 추출 클래스 초기화
        self.model, self.device = self._init_detection_model(args)
        self.avg_pool = nn.AvgPool2d(args.roi_feat_size)

    def _get_feats(self, args):
        # TalkNet을 사용할 경우 TalkNet 기능으로, 그렇지 않을 경우 Annotated 기능으로 비디오 특징 추출
        if args.use_TalkNet:
            self.bbox_feats = self._get_TalkNet_features(args)
        
        # TODO: TalkNet을 사용하지 않을 경우가 어떤건지 확인 필요
        else:
            self.bbox_feats = self._get_Annotated_features(args)
            
    def _save_feats(self, args):
        # 추출된 특징을 파일로 저장
        video_feats_path = os.path.join(args.video_data_path, args.video_feats_path)

        with open(video_feats_path, 'wb') as f:
            pickle.dump(self.bbox_feats, f)   

    def _init_detection_model(self, args):
        # 객체 감지 모델 초기화, mmdetection과 관련된 코드
        model = init_detector(args.detection_config_path, args.detection_checkpoint_path, device='cuda:0')
        device = next(model.parameters()).device  # model device
        return model, device

    def _get_TalkNet_features(self, args):
        '''
        TalkNet 모델 주석을 사용하여 비디오 클립에서 특징을 추출하는 함수입니다.

        입력: 
            args.TalkNet_speaker_path: TalkNet 화자 데이터가 저장된 경로입니다.

        출력:
        비디오 특징의 형식은 다음과 같습니다.
        {
            'video_clip_id_a':[frame_a_feat, frame_b_feat, ..., frame_N_feat],
            'video_clip_id_b':[xxx]
        }
        '''

        video_feats = {}  # 비디오 특징을 저장할 사전을 초기화합니다.
        error_cnt = 0  # 발생한 오류를 세는 카운터입니다.
        error_path = 0  # 경로 관련 오류를 세는 카운터입니다.
        
        # 지정된 TalkNet 화자 경로에 있는 각 비디오 클립을 반복합니다.
        for video_clip_name in tqdm(os.listdir(args.TalkNet_speaker_path), desc='Video'):
            frames_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pyframes')
            bestperson_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pywork', 'best_persons.npy')
            
            # bestperson_path가 존재하지 않으면 비디오 클립을 건너뜁니다.
            if not os.path.exists(bestperson_path):
                error_path += 1
                continue

            # bounding box 정보를 numpy 배열로 불러옵니다.
            bestpersons = np.load(bestperson_path)
            
            # 각 프레임에 대한 bounding box를 순회하며 특징 추출
            for frame, bbox in tqdm(enumerate(bestpersons), desc='Frame'):
                # 빈 bounding box인 경우 건너뜁니다.
                if (bbox[0] == 0) and (bbox[1] == 0) and (bbox[2] == 0) and (bbox[3] == 0):
                    error_cnt += 1
                    continue
                
                # 프레임 이름을 설정합니다.
                frame_name = str('%06d' % frame)
                frame_path = os.path.join(frames_path, frame_name + '.jpg')

                # bounding box를 리스트로 변환하고 roi 리스트를 생성합니다.
                roi = bbox.tolist()              
                roi.insert(0, 0.)

                # bounding box 기반으로 특징을 추출합니다.
                bbox_feat = self._extract_roi_feats(self.model, self.device, frame_path, roi)
                # 평균 풀링을 적용합니다.
                bbox_feat = self._average_pooling(bbox_feat)
                # numpy 배열로 변환합니다.
                bbox_feat = bbox_feat.detach().cpu().numpy()
                
                # 추출된 특징을 video_feats에 저장합니다.
                if video_clip_name not in video_feats.keys():
                    video_feats[video_clip_name] = [bbox_feat]
                else:
                    video_feats[video_clip_name].append(bbox_feat)
            
            # 오류 수를 출력합니다.
            print('오류 발생한 주석의 수: {}'.format(error_cnt))
            print('오류 발생한 경로의 수: {}'.format(error_path))
            
        return video_feats

            
    def _get_Annotated_features(self, args):

        '''
        Input: 
            args.video_data_path 
            args.speaker_annotation_path
            args.frames_path

        Output:
        The format of video features
        {
            'video_clip_id_a':[frame_a_feat, frame_b_feat, ..., frame_N_feat],
            'video_clip_id_b':[xxx]
        }
        '''

        speaker_annotation_path = os.path.join(args.video_data_path, args.speaker_annotation_path)
        speaker_annotations = json.load(open(speaker_annotation_path, 'r'))

        video_feats = {}
        error_cnt = 0

        try:
            for key in tqdm(speaker_annotations.keys(), desc = 'Frame'):
                
                if 'bbox' not in speaker_annotations[key].keys():
                    error_cnt += 1
                    continue
                
                roi = speaker_annotations[key]['bbox'][:4]
                roi.insert(0, 0.)

                frame_name = '_'.join(key.strip('.jpg').split('_')[:-1])
                frame_path = os.path.join(args.frames_path, frame_name + '.jpg')
                
                bbox_feat = self._extract_roi_feats(self.model, self.device, frame_path, roi)
                bbox_feat = self._average_pooling(bbox_feat)
                bbox_feat = bbox_feat.detach().cpu().numpy()
                
                video_clip_name = '_'.join(key.strip('.jpg').split('_')[:-2])

                if video_clip_name not in video_feats.keys():
                    video_feats[video_clip_name] = [bbox_feat]
                
                else:
                    video_feats[video_clip_name].append(bbox_feat)

        except Exception as e:
                print(e)

        print('The number of error annotations is {}'.format(error_cnt))

        return video_feats         

    
    def _extract_roi_feats(self, model, device, file_path, roi):
        
        roi = torch.tensor([roi]).to(device)
        cfg = model.cfg
        # prepare data
        data = dict(img_info=dict(filename = file_path), img_prefix=None)
        # build the data pipeline
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]

        img = data['img'][0]
        x = model.extract_feat(img)

        bbox_feat = model.roi_head.bbox_roi_extractor(
            x[:model.roi_head.bbox_roi_extractor.num_inputs], roi)
        
        return bbox_feat

    def _average_pooling(self, x):
        """
        Args:
        x: dtype: numpy.ndarray
        """
        x = self.avg_pool(x)
        x = x.flatten(1)
        return x

if __name__ == '__main__':

    args = parse_arguments()
    
    args.use_TalkNet = True
    video_data = VideoFeature(args)
    video_data._get_feats(args)
    video_data._save_feats(args)
    
