#!/usr/bin/env python3
"""
안정화된 임베딩 모델 평가 스크립트
메모리 효율성과 에러 처리 강화
TestCase 특화 평가로 업데이트 (Tech 지표 개선)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import time
import warnings
import gc
import torch
warnings.filterwarnings('ignore')

def clear_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_sfr_model(model_name):
    """Jina, SFR 모델 전용 로더"""
    try:
        print(f"  🔄 SFR 모델 로딩: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
        
        # GPU 사용 가능하면 GPU로 이동
        if torch.cuda.is_available():
            model = model.cuda()
            print("  📱 GPU로 모델 이동 완료")
        
        return model, tokenizer
    except Exception as e:
        print(f"  ❌ SFR 모델 로딩 실패: {e}")
        return None, None

def encode_with_sfr(texts, model, tokenizer, batch_size=1):
    """SFR 모델을 사용한 임베딩 생성 (모델별 최적화)"""
    try:
        import torch.nn.functional as F
        
        # 모델별 설정
        model_name = getattr(model, 'name_or_path', str(model))
        
        if 'Code-2B_R' in model_name:
            # Code-2B_R 모델: encode_corpus 메서드 사용
            print("    🔧 Code-2B_R 모델용 인코딩")
            try:
                # Code 모델은 특별한 인코딩 메서드를 제공
                embeddings = model.encode_corpus(texts, max_length=4096)
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
                return embeddings
            except AttributeError:
                print("    ⚠️ encode_corpus 메서드 없음, 기본 방식 사용")
                # 기본 방식으로 폴백
                pass
        
        # 기본 SFR 임베딩 생성 (Mistral 등)
        def last_token_pool(last_hidden_states, attention_mask):
            """Last token pooling"""
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
        embeddings = []
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 토크나이징
                max_length = 32768 if 'Code-2B_R' in model_name else 4096
                inputs = tokenizer(batch_texts, 
                                 return_tensors="pt", 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=max_length)
                
                # GPU로 이동 (모델이 GPU에 있는 경우)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 모델 추론
                outputs = model(**inputs)
                
                # Last token pooling 적용
                batch_embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                
                # 정규화
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                # CPU로 이동하고 numpy로 변환
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)
                
                # 중간 메모리 정리
                if i % 5 == 0:
                    clear_memory()
        
        # 모든 배치 결합
        return np.vstack(embeddings)
        
    except Exception as e:
        print(f"    SFR 임베딩 생성 오류: {e}")
        return None

def evaluate_single_model(embedding_list, model_name, max_texts=None, batch_size=2):
    """
    단일 모델을 안전하게 평가합니다.
    
    Args:
        embedding_list: 이슈별 텍스트 리스트
        model_name: 모델명
        max_texts: 최대 처리할 텍스트 수 (None이면 전체)
        batch_size: 배치 크기 (작게 설정)
    """
    
    print(f"\n🔬 {model_name} 모델 평가 시작...")
    
    # 모든 텍스트 수집
    all_texts = []
    text_to_issue = {}
    
    for issue_idx, issue_text_list in enumerate(embedding_list):
        issue_id = f"issue_{issue_idx}"
        
        if isinstance(issue_text_list, list):
            for text_idx, text in enumerate(issue_text_list):
                if text and isinstance(text, str) and text.strip():
                    all_texts.append(text)
                    text_to_issue[len(all_texts)-1] = f"{issue_id}_text_{text_idx}"
                    
                    # 최대 텍스트 수 제한
                    if max_texts and len(all_texts) >= max_texts:
                        break
            if max_texts and len(all_texts) >= max_texts:
                break

    if not all_texts:
        print("  ❌ 평가할 텍스트가 없습니다.")
        return None
    
    print(f"  📝 {len(all_texts)}개 텍스트 처리 예정 (전체 임베딩)")
    print(f"  🎯 TestCase 평가는 처음 3개 이슈만 사용")
    
    try:
        # 메모리 정리
        clear_memory()
        
        # 모델 로드 시도
        print("  🔄 모델 로딩 중...")
        
        # SFR 모델 특별 처리
        if 'SFR-Embedding' in model_name or 'sfr-embedding' in model_name.lower() or 'jina-embedding' in model_name:
            print("  🎯 SFR 모델 감지 - 특별 처리 모드")
            sfr_model, sfr_tokenizer = load_sfr_model(model_name)
            
            if sfr_model is None or sfr_tokenizer is None:
                print("  ❌ SFR 모델 로딩 실패")
                return None
            
            # SFR 모델로 임베딩 생성
            print("  🔄 SFR 임베딩 생성 중...")
            start_time = time.time()
            
            # 청크 단위로 처리 (메모리 절약)
            chunk_size = min(20, len(all_texts))  # SFR은 20개씩 처리
            all_embeddings = []
            
            for i in range(0, len(all_texts), chunk_size):
                chunk_texts = all_texts[i:i+chunk_size]
                print(f"    처리 중: {i+1}-{min(i+chunk_size, len(all_texts))}/{len(all_texts)}")
                
                chunk_embeddings = encode_with_sfr(chunk_texts, sfr_model, sfr_tokenizer, batch_size=1)
                if chunk_embeddings is not None:
                    all_embeddings.append(chunk_embeddings)
                
                # 중간 메모리 정리
                if i % 20 == 0:
                    clear_memory()
            
            if not all_embeddings:
                print("  ❌ SFR 임베딩 생성 실패")
                return None
            
            # 모든 임베딩 합치기
            embeddings = np.vstack(all_embeddings)
            embedding_time = time.time() - start_time
            
            # SFR 모델 정리
            del sfr_model
            del sfr_tokenizer
            clear_memory()
            
        else:
            # 기존 SentenceTransformer 방식 (다국어 모델들)
            # 모델별 로딩 전략
            if 'jina' in model_name.lower():
                model = SentenceTransformer(model_name, trust_remote_code=True)
            elif 'bge' in model_name.lower():
                model = SentenceTransformer(model_name)
            elif 'multilingual-e5' in model_name.lower():
                model = SentenceTransformer(model_name)
            elif 'e5-mistral' in model_name.lower():
                model = SentenceTransformer(model_name, trust_remote_code=True)
            else:
                model = SentenceTransformer(model_name, trust_remote_code=True)
                
            print(f"  ✅ 다국어 모델 로딩 완료 (배치 크기: {batch_size})")
            
            # 임베딩 생성
            print("  🔄 다국어 임베딩 생성 중...")
            start_time = time.time()
            
            # 청크 단위로 처리 (메모리 절약)
            chunk_size = min(50, len(all_texts))  # 50개씩 처리
            all_embeddings = []
            
            for i in range(0, len(all_texts), chunk_size):
                chunk_texts = all_texts[i:i+chunk_size]
                print(f"    처리 중: {i+1}-{min(i+chunk_size, len(all_texts))}/{len(all_texts)}")
                
                chunk_embeddings = model.encode(
                    chunk_texts, 
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # 다국어 모델에서 정규화 중요
                )
                all_embeddings.append(chunk_embeddings)
                
                # 중간 메모리 정리
                if i % 100 == 0:
                    clear_memory()
            
            # 모든 임베딩 합치기
            embeddings = np.vstack(all_embeddings)
            embedding_time = time.time() - start_time
            
            # 모델 정리
            del model
            clear_memory()
        
        print(f"  ✅ 임베딩 생성 완료 ({embedding_time:.2f}초)")
        
        # 평가 결과 저장
        evaluation_results = {
            'model_name': model_name,
            'total_texts': len(all_texts),
            'total_issues': len(embedding_list),
            'embedding_dimension': embeddings.shape[1],
            'embedding_time': embedding_time,
            'texts_per_second': len(all_texts) / embedding_time if embedding_time > 0 else 0,
            'batch_size_used': batch_size
        }
        
        # 품질 평가 (전체 데이터)
        print("  📊 품질 평가 중...")
        quality_metrics = evaluate_embedding_quality(embeddings)
        evaluation_results.update(quality_metrics)
        
        # 유사도 분석 (전체 데이터)
        print("  🔍 유사도 분석 중...")
        similarity_metrics = analyze_similarity_distribution(embeddings, text_to_issue)
        evaluation_results.update(similarity_metrics)
        
        # 클러스터링 평가 (전체 데이터)
        print("  🎯 클러스터링 평가 중...")
        clustering_metrics = evaluate_clustering_performance(embeddings, text_to_issue)
        evaluation_results.update(clustering_metrics)
        
        # TestCase 특화 평가 추가 (처음 3개 이슈만 사용)
        print("  🎯 TestCase 특화 평가 중...")
        testcase_metrics = evaluate_testcase_performance(embeddings, text_to_issue, all_texts, embedding_list)
        evaluation_results.update(testcase_metrics)
        
        # 메모리 정리
        del embeddings
        clear_memory()
        
        print(f"  ✅ {model_name} 평가 완료!")
        return evaluation_results
        
    except Exception as e:
        print(f"  ❌ {model_name} 평가 실패: {str(e)}")
        clear_memory()
        return None

def evaluate_embedding_quality(embeddings):
    """임베딩 품질 평가 (메모리 효율적)"""
    try:
        # 벡터 크기 분포
        norms = np.linalg.norm(embeddings, axis=1)
        
        # 차원별 분산 (샘플링)
        if embeddings.shape[1] > 1000:
            # 차원이 너무 크면 샘플링
            sample_dims = np.random.choice(embeddings.shape[1], 1000, replace=False)
            dim_variances = np.var(embeddings[:, sample_dims], axis=0)
        else:
            dim_variances = np.var(embeddings, axis=0)
        
        # 임베딩 간 평균 거리 (샘플링)
        if len(embeddings) > 100:
            # 너무 많으면 샘플링
            sample_indices = np.random.choice(len(embeddings), 100, replace=False)
            sample_embeddings = embeddings[sample_indices]
            distances = euclidean_distances(sample_embeddings)
        else:
            distances = euclidean_distances(embeddings)
            
        avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
        
        return {
            'vector_norm_mean': float(np.mean(norms)),
            'vector_norm_std': float(np.std(norms)),
            'dimension_variance_mean': float(np.mean(dim_variances)),
            'dimension_variance_std': float(np.std(dim_variances)),
            'average_euclidean_distance': float(avg_distance),
            'embedding_density': float(1.0 / avg_distance if avg_distance > 0 else 0)
        }
    except Exception as e:
        print(f"    품질 평가 오류: {e}")
        return {
            'vector_norm_mean': 0.0, 'vector_norm_std': 0.0,
            'dimension_variance_mean': 0.0, 'dimension_variance_std': 0.0,
            'average_euclidean_distance': 0.0, 'embedding_density': 0.0
        }

def analyze_similarity_distribution(embeddings, text_to_issue):
    """유사도 분석 (개선된 디버깅 버전)"""
    try:
        print(f"    🔍 유사도 분석 디버깅: 임베딩 shape = {embeddings.shape}")
        
        # 임베딩 정규화
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(embeddings, norm='l2')
        
        # 코사인 유사도 계산 (청크 단위)
        if len(embeddings) > 100:
            # 샘플링
            sample_indices = np.random.choice(len(embeddings), 100, replace=False)
            sample_embeddings = normalized_embeddings[sample_indices]
            sample_text_to_issue = {i: text_to_issue[sample_indices[i]] for i in range(len(sample_indices))}
            similarities = cosine_similarity(sample_embeddings)
            current_text_to_issue = sample_text_to_issue
        else:
            similarities = cosine_similarity(normalized_embeddings)
            current_text_to_issue = text_to_issue
        
        print(f"    📊 유사도 매트릭스 크기: {similarities.shape}")
        print(f"    📈 유사도 범위: {similarities.min():.4f} ~ {similarities.max():.4f}")
        
        # 자기 자신과의 유사도 제외
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        
        # 같은 이슈 내 텍스트들 간 유사도
        same_issue_similarities = []
        different_issue_similarities = []
        
        issue_pair_count = defaultdict(int)
        
        for i in range(len(similarities)):
            for j in range(i+1, len(similarities)):
                sim = similarities[i][j]
                issue_i = current_text_to_issue[i].split('_')[0]
                issue_j = current_text_to_issue[j].split('_')[0]
                
                if issue_i == issue_j:
                    same_issue_similarities.append(sim)
                    issue_pair_count['same'] += 1
                else:
                    different_issue_similarities.append(sim)
                    issue_pair_count['different'] += 1
        
        print(f"    🔗 같은 이슈 내 쌍: {issue_pair_count['same']}개")
        print(f"    🔗 다른 이슈 간 쌍: {issue_pair_count['different']}개")
        
        # 안전한 평균 계산
        same_issue_mean = float(np.mean(same_issue_similarities)) if same_issue_similarities else 0.0
        different_issue_mean = float(np.mean(different_issue_similarities)) if different_issue_similarities else 0.0
        ratio = same_issue_mean / different_issue_mean if different_issue_mean > 0 else 0.0
        
        print(f"    📊 같은 이슈 내 평균 유사도: {same_issue_mean:.4f}")
        print(f"    📊 다른 이슈 간 평균 유사도: {different_issue_mean:.4f}")
        print(f"    📊 유사도 비율: {ratio:.4f}")
        
        # 임베딩 분포 확인
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        print(f"    📏 임베딩 크기 분포: {embedding_norms.min():.4f} ~ {embedding_norms.max():.4f} (평균: {embedding_norms.mean():.4f})")
        
        return {
            'similarity_mean': float(np.mean(upper_triangle)),
            'similarity_std': float(np.std(upper_triangle)),
            'similarity_min': float(np.min(upper_triangle)),
            'similarity_max': float(np.max(upper_triangle)),
            'same_issue_similarity_mean': same_issue_mean,
            'different_issue_similarity_mean': different_issue_mean,
            'intra_vs_inter_ratio': ratio,
            'same_issue_pairs': issue_pair_count.get('same', 0),
            'different_issue_pairs': issue_pair_count.get('different', 0),
            'embedding_norm_mean': float(embedding_norms.mean()),
            'embedding_norm_std': float(embedding_norms.std())
        }
    except Exception as e:
        print(f"    ❌ 유사도 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'similarity_mean': 0.0, 'similarity_std': 0.0,
            'similarity_min': 0.0, 'similarity_max': 0.0,
            'same_issue_similarity_mean': 0.0, 'different_issue_similarity_mean': 0.0,
            'intra_vs_inter_ratio': 0.0, 'same_issue_pairs': 0, 'different_issue_pairs': 0,
            'embedding_norm_mean': 0.0, 'embedding_norm_std': 0.0
        }

def evaluate_clustering_performance(embeddings, text_to_issue):
    """클러스터링 성능 평가 (개선된 버전)"""
    try:
        print(f"    🔍 클러스터링 디버깅: 임베딩 shape = {embeddings.shape}")
        
        # 이슈별로 그룹화
        issue_groups = defaultdict(list)
        for idx, issue_info in text_to_issue.items():
            issue_id = issue_info.split('_')[0]
            issue_groups[issue_id].append(idx)
        
        num_issues = len(issue_groups)
        print(f"    📊 이슈 수: {num_issues}, 텍스트 수: {len(embeddings)}")
        
        # 각 이슈별 텍스트 개수 출력
        for issue_id, indices in issue_groups.items():
            print(f"      - {issue_id}: {len(indices)}개 텍스트")
        
        if num_issues < 2 or len(embeddings) < 2:
            print(f"    ⚠️ 클러스터링 불가: 이슈 수({num_issues}) 또는 임베딩 수({len(embeddings)})가 부족")
            return {
                'clustering_score': 0.0, 
                'best_clustering_score': 0.0,
                'optimal_clusters': 0,
                'actual_issues': num_issues,
                'debug_info': f"issues={num_issues}, embeddings={len(embeddings)}"
            }
        
        # 임베딩 정규화 (코사인 유사도 기반 클러스터링을 위해)
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(embeddings, norm='l2')
        
        # 여러 클러스터 수로 테스트
        best_score = -1.0
        best_n_clusters = 2
        
        for n_clusters in range(2, min(num_issues + 1, len(embeddings), 11)):
            try:
                print(f"    🎯 클러스터 수 {n_clusters} 테스트 중...")
                
                kmeans = KMeans(
                    n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10,
                    max_iter=300
                )
                cluster_labels = kmeans.fit_predict(normalized_embeddings)
                
                # 클러스터 결과 확인
                unique_clusters = len(set(cluster_labels))
                print(f"      - 실제 생성된 클러스터 수: {unique_clusters}")
                
                if unique_clusters > 1 and unique_clusters == n_clusters:
                    silhouette = silhouette_score(normalized_embeddings, cluster_labels)
                    print(f"      - 실루엣 점수: {silhouette:.4f}")
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_n_clusters = n_clusters
                else:
                    print(f"      - 클러스터링 실패 (유니크 클러스터: {unique_clusters})")
                    
            except Exception as cluster_error:
                print(f"      - 클러스터 {n_clusters} 오류: {cluster_error}")
                continue
        
        # 추가 평가: 이슈 기반 평가 (실제 레이블과의 비교)
        issue_based_score = 0.0
        try:
            # 실제 이슈 레이블 생성
            true_labels = []
            for idx in range(len(embeddings)):
                issue_info = text_to_issue[idx]
                issue_id = issue_info.split('_')[0]
                issue_num = int(issue_id.split('_')[1]) if 'issue_' in issue_id else 0
                true_labels.append(issue_num)
            
            if len(set(true_labels)) > 1:
                issue_based_score = silhouette_score(normalized_embeddings, true_labels)
                print(f"    📋 실제 이슈 기반 점수: {issue_based_score:.4f}")
        except Exception as e:
            print(f"    ⚠️ 이슈 기반 평가 실패: {e}")
        
        final_score = max(best_score, issue_based_score)
        
        print(f"    ✅ 최종 클러스터링 점수: {final_score:.4f}")
        
        return {
            'clustering_score': float(final_score),
            'best_clustering_score': float(best_score),
            'issue_based_score': float(issue_based_score),
            'optimal_clusters': int(best_n_clusters),
            'actual_issues': int(num_issues),
            'debug_info': f"best_score={best_score:.4f}, issue_score={issue_based_score:.4f}"
        }
        
    except Exception as e:
        print(f"    ❌ 클러스터링 전체 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'clustering_score': 0.0, 
            'best_clustering_score': 0.0,
            'issue_based_score': 0.0,
            'optimal_clusters': 0, 
            'actual_issues': 0,
            'debug_info': f"error: {str(e)}"
        }

def evaluate_testcase_performance(embeddings, text_to_issue, all_texts, embedding_list=None):
    """TestCase 특화 성능 평가 (3번째 리스트까지만 사용)"""
    
    print(f"\n🎯 TestCase 특화 성능 평가 (처음 3개 이슈 기준)")
    
    # TestCase 평가용 데이터 필터링 (처음 3개 이슈만)
    testcase_indices = []
    testcase_texts = []
    testcase_text_to_issue = {}
    
    if embedding_list and len(embedding_list) >= 3:
        # 처음 3개 이슈의 인덱스 찾기
        current_idx = 0
        for issue_idx in range(3):  # 0, 1, 2번째 이슈만
            if issue_idx < len(embedding_list):
                issue_text_list = embedding_list[issue_idx]
                if isinstance(issue_text_list, list):
                    for text_idx, text in enumerate(issue_text_list):
                        if text and isinstance(text, str) and text.strip():
                            testcase_indices.append(current_idx)
                            testcase_texts.append(text)
                            testcase_text_to_issue[len(testcase_texts)-1] = f"issue_{issue_idx}_text_{text_idx}"
                            current_idx += 1
                        else:
                            current_idx += 1
                else:
                    current_idx += 1
            else:
                break
    else:
        # embedding_list가 없으면 전체 데이터에서 처음 3개 이슈 추정
        issue_counts = {}
        for idx, issue_info in text_to_issue.items():
            issue_id = issue_info.split('_')[0]
            if issue_id not in issue_counts:
                issue_counts[issue_id] = 0
            issue_counts[issue_id] += 1
        
        # 처음 3개 이슈 선택
        first_three_issues = list(sorted(issue_counts.keys()))[:3]
        
        for idx, issue_info in text_to_issue.items():
            issue_id = issue_info.split('_')[0]
            if issue_id in first_three_issues:
                testcase_indices.append(idx)
                testcase_texts.append(all_texts[idx])
                testcase_text_to_issue[len(testcase_texts)-1] = issue_info
    
    if not testcase_indices:
        print(f"    ⚠️ TestCase 평가용 데이터가 없습니다")
        return {
            'testcase_functional_scores': {},
            'testcase_multilingual_score': 0.0,
            'testcase_technical_consistency': 0.0,
            'testcase_overall_score': 0.0,
            'functional_group_sizes': {},
            'testcase_data_used': 0
        }
    
    # TestCase 평가용 임베딩 추출
    testcase_embeddings = embeddings[testcase_indices]
    
    print(f"    📊 TestCase 평가 데이터: {len(testcase_texts)}개 (처음 3개 이슈)")
    
    # 기능별 그룹 정의 (TestCase 데이터만으로)
    functional_groups = {
        "시간_동기화": [],
        "카드_인식": [], 
        "스케줄_관리": []
    }
    
    # 텍스트 내용 기반으로 그룹 분류
    for idx, text in enumerate(testcase_texts):
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["time synchronization", "시간 동기화", "dhcp", "time zone", "biostar server"]):
            functional_groups["시간_동기화"].append(idx)
        elif any(keyword in text_lower for keyword in ["hid prox", "fsk", "wiegand", "카드", "card format"]):
            functional_groups["카드_인식"].append(idx)
        elif any(keyword in text_lower for keyword in ["schedule", "스케줄", "holiday", "time slot", "weekly", "daily"]):
            functional_groups["스케줄_관리"].append(idx)
    
    print(f"    📊 TestCase 기능별 그룹 크기:")
    for group_name, indices in functional_groups.items():
        print(f"      - {group_name}: {len(indices)}개")
    
    # 1. 기능별 클러스터링 점수
    functional_scores = {}
    for group_name, indices in functional_groups.items():
        if len(indices) >= 2:
            group_embeddings = testcase_embeddings[indices]
            from sklearn.preprocessing import normalize
            normalized_embeddings = normalize(group_embeddings, norm='l2')
            
            # 그룹 내 평균 유사도
            similarities = cosine_similarity(normalized_embeddings)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            intra_similarity = np.mean(upper_triangle)
            functional_scores[group_name] = intra_similarity
            
            print(f"    🔍 {group_name} 그룹 내 유사도: {intra_similarity:.4f}")
    
    # 2. 다국어 키워드 매칭 성능 (TestCase 데이터만)
    korean_keywords = ["스케줄", "설정", "인증", "시간", "카드", "장치"]
    english_keywords = ["schedule", "setting", "authentication", "time", "card", "device"]
    
    multilingual_score = 0.0
    keyword_matches = 0
    
    for kr_word, en_word in zip(korean_keywords, english_keywords):
        kr_indices = [i for i, text in enumerate(testcase_texts) if kr_word in text]
        en_indices = [i for i, text in enumerate(testcase_texts) if en_word.lower() in text.lower()]
        
        if kr_indices and en_indices:
            # 한국어-영어 키워드 간 유사도
            kr_embeddings = testcase_embeddings[kr_indices]
            en_embeddings = testcase_embeddings[en_indices]
            
            cross_similarities = cosine_similarity(kr_embeddings, en_embeddings)
            avg_cross_similarity = np.mean(cross_similarities)
            multilingual_score += avg_cross_similarity
            keyword_matches += 1
    
    if keyword_matches > 0:
        multilingual_score /= keyword_matches
        print(f"    🌏 다국어 키워드 매칭 점수: {multilingual_score:.4f}")
    
    # 3. 기술 용어 일관성 점수 (순수 기술 용어만 사용)
    technical_terms = [
        "Time Slot",           # 스케줄 관리 기술 용어
        "Holiday",             # 스케줄 관리 기술 용어  
        "Device",              # 하드웨어 기술 용어
        "DHCP",                # 네트워크 기술 용어
        "Time Synchronization", # 시간 동기화 기술 용어
        "Wiegand",             # 카드 인식 기술 용어
        "Authentication",      # 인증 기술 용어
        "Schedule"             # 스케줄 기술 용어
    ]
    
    technical_consistency = 0.0
    term_matches = 0
    
    for term in technical_terms:
        term_indices = [i for i, text in enumerate(testcase_texts) if term in text]
        if len(term_indices) >= 2:
            term_embeddings = testcase_embeddings[term_indices]
            similarities = cosine_similarity(term_embeddings)
            upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
            term_consistency = np.mean(upper_triangle)
            technical_consistency += term_consistency
            term_matches += 1
    
    if term_matches > 0:
        technical_consistency /= term_matches
        print(f"    🔧 기술 용어 일관성 점수: {technical_consistency:.4f}")
    
    # 4. 전체 TestCase 특화 점수 계산
    testcase_score = 0.0
    score_components = 0
    
    if functional_scores:
        avg_functional_score = np.mean(list(functional_scores.values()))
        testcase_score += avg_functional_score * 0.4  # 40% 가중치
        score_components += 1
        
    if multilingual_score > 0:
        testcase_score += multilingual_score * 0.3  # 30% 가중치
        score_components += 1
        
    if technical_consistency > 0:
        testcase_score += technical_consistency * 0.3  # 30% 가중치
        score_components += 1
    
    if score_components > 0:
        testcase_score = testcase_score  # 이미 가중평균됨
    
    print(f"    ⭐ TestCase 특화 점수 (3개 이슈 기준): {testcase_score:.4f}")
    
    return {
        'testcase_functional_scores': functional_scores,
        'testcase_multilingual_score': multilingual_score,
        'testcase_technical_consistency': technical_consistency,
        'testcase_overall_score': testcase_score,
        'functional_group_sizes': {k: len(v) for k, v in functional_groups.items()},
        'testcase_data_used': len(testcase_texts)
    }

def evaluate_models_safely(embedding_list, model_names=None, max_texts_per_model=100):
    """
    안전하게 여러 모델을 평가합니다.
    
    Args:
        embedding_list: 이슈별 텍스트 리스트
        model_names: 평가할 모델명 리스트
        max_texts_per_model: 모델당 최대 처리할 텍스트 수
    """
    
    if model_names is None:
        # 한국어-영어 혼합 환경 기본 모델들
        model_names = [
            'sentence-transformers/all-MiniLM-L6-v2',  # 영어 기준선
            'BAAI/bge-m3',                              # 다국어 최고 성능
            'jinaai/jina-embeddings-v3',                # 한국어 특화
            'Salesforce/SFR-Embedding-Mistral'         # SFR 모델
        ]
    
    print(f"🚀 다국어 모드로 {len(model_names)}개 모델 평가 시작")
    print(f"📊 모델당 최대 {max_texts_per_model}개 텍스트 처리")
    print(f"🌏 한국어-영어 혼합 데이터 최적화")
    
    results = {}
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n[{i}/{len(model_names)}] {model_name}")
        print("-" * 60)
        
        # 다국어 모델의 경우 배치 크기 조정
        if 'bge-m3' in model_name or 'e5-mistral' in model_name:
            batch_size = 1  # 큰 다국어 모델은 배치 크기 줄임
            max_texts = min(max_texts_per_model, 100)
        elif 'jina-embeddings-v3' in model_name or 'Salesforce/SFR-Embedding-Mistral' in model_name:
            batch_size = 2  # 효율적인 모델
            max_texts = max_texts_per_model
        else:
            batch_size = 4  # 작은 모델
            max_texts = max_texts_per_model
        
        result = evaluate_single_model(
            embedding_list, 
            model_name, 
            max_texts=max_texts,
            batch_size=batch_size
        )
        
        results[model_name] = result
        
        # 각 모델 후 메모리 정리
        clear_memory()
        time.sleep(3)  # 다국어 모델은 조금 더 대기
    
    # 결과 출력
    print_results_table(results)
    
    return results

def print_results_table(results):
    """결과 테이블 출력 (Functional, Multi, Tech 3개 지표)"""
    
    print(f"\n{'='*130}")
    print("🏆 TestCase 임베딩 모델 종합 평가 결과")
    print(f"{'='*130}")
    
    # 성공한 모델들만 필터링
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if not successful_results:
        print("❌ 성공적으로 평가된 모델이 없습니다.")
        return
    
    # 개선된 헤더 (TestCase 제거, Functional 추가)
    print(f"{'Model':<35} {'Status':<8} {'Texts':<6} {'Speed':<10} {'Dim':<6} {'Functional':<10} {'Multi':<8} {'Tech':<8}")
    print("-" * 130)
    
    # 각 모델 결과
    for model_name, result in results.items():
        if result is None:
            row = f"{model_name:<35} {'FAILED':<8} {'-':<6} {'-':<10} {'-':<6} {'-':<10} {'-':<8} {'-':<8}"
        else:
            status = "SUCCESS"
            texts = str(result.get('total_texts', 0))
            speed = f"{result.get('texts_per_second', 0):.1f}t/s"
            dimension = str(result.get('embedding_dimension', 0))
            
            # Functional 점수 계산 (기능별 유사도의 평균)
            functional_scores = result.get('testcase_functional_scores', {})
            if functional_scores:
                functional = f"{np.mean(list(functional_scores.values())):.3f}"
            else:
                functional = "0.000"
            
            multilingual = f"{result.get('testcase_multilingual_score', 0):.3f}"
            technical = f"{result.get('testcase_technical_consistency', 0):.3f}"
            
            row = f"{model_name:<35} {status:<8} {texts:<6} {speed:<10} {dimension:<6} {functional:<10} {multilingual:<8} {technical:<8}"
        
        print(row)
    
    print(f"{'='*130}")
    print("📊 지표 설명:")
    print("  • Speed: 임베딩 생성 속도 (텍스트/초)")
    print("  • Dim: 임베딩 벡터 차원 수")
    print("  • Functional: 기능별 그룹 내 유사도 (0~1)")
    print("  • Multi: 다국어 키워드 매칭 점수 (0~1)")
    print("  • Tech: 기술 용어 일관성 점수 (0~1)")
    
    # 성능 vs 효율성 분석
    print(f"\n⚡ 성능 vs 효율성 분석:")
    for model_name, result in successful_results.items():
        speed = result.get('texts_per_second', 0)
        functional_scores = result.get('testcase_functional_scores', {})
        functional = np.mean(list(functional_scores.values())) if functional_scores else 0
        dimension = result.get('embedding_dimension', 0)
        
        efficiency_ratio = speed / (dimension / 1000) if dimension > 0 else 0
        
        print(f"  🔬 {model_name}:")
        print(f"    - 성능: Functional {functional:.3f} | Multi {result.get('testcase_multilingual_score', 0):.3f} | Tech {result.get('testcase_technical_consistency', 0):.3f}")
        print(f"    - 효율성: {speed:.1f}t/s | {dimension}차원 | 효율비 {efficiency_ratio:.2f}")
    
    # 상세 분석
    print(f"\n📊 TestCase 특화 상세 분석:")
    for model_name, result in successful_results.items():
        functional_scores = result.get('testcase_functional_scores', {})
        functional_avg = np.mean(list(functional_scores.values())) if functional_scores else 0
        
        print(f"\n🔬 {model_name}:")
        print(f"  • Functional 점수: {functional_avg:.4f}")
        
        # 기능별 점수
        if functional_scores:
            print(f"  • 기능별 유사도:")
            for func, score in functional_scores.items():
                print(f"    - {func}: {score:.4f}")
        
        # 그룹 크기
        group_sizes = result.get('functional_group_sizes', {})
        if group_sizes:
            print(f"  • 기능별 테스트 수:")
            for func, size in group_sizes.items():
                print(f"    - {func}: {size}개")
        
        print(f"  • Multi 점수: {result.get('testcase_multilingual_score', 0):.4f}")
        print(f"  • Tech 점수: {result.get('testcase_technical_consistency', 0):.4f}")
        print(f"  • TestCase 평가 데이터: {result.get('testcase_data_used', 0)}개 (처음 3개 이슈)")
        print(f"  • 전체 임베딩 데이터: {result.get('total_texts', 0)}개")
        print(f"  • 임베딩 시간: {result.get('embedding_time', 0):.2f}초")
    
    # 종합 추천
    print(f"\n🏆 종합 추천 (지표별):")
    
    # 1. Functional 기준
    functional_rankings = []
    for model_name, result in successful_results.items():
        functional_scores = result.get('testcase_functional_scores', {})
        if functional_scores:
            functional_avg = np.mean(list(functional_scores.values()))
            functional_rankings.append((model_name, functional_avg))
    
    if functional_rankings:
        functional_rankings.sort(key=lambda x: x[1], reverse=True)
        print(f"  🎯 Functional (기능별 유사도) 순:")
        for i, (model, score) in enumerate(functional_rankings[:3], 1):
            print(f"    {i}. {model}: {score:.4f}")
    
    # 2. Multi 기준
    by_multilingual = sorted(successful_results.items(), 
                            key=lambda x: x[1].get('testcase_multilingual_score', 0), reverse=True)
    if by_multilingual:
        print(f"  🌏 Multi (다국어 매칭) 순:")
        for i, (model, result) in enumerate(by_multilingual[:3], 1):
            score = result.get('testcase_multilingual_score', 0)
            print(f"    {i}. {model}: {score:.4f}")
    
    # 3. Tech 기준
    by_technical = sorted(successful_results.items(), 
                         key=lambda x: x[1].get('testcase_technical_consistency', 0), reverse=True)
    if by_technical:
        print(f"  🔧 Tech (기술 용어 일관성) 순:")
        for i, (model, result) in enumerate(by_technical[:3], 1):
            score = result.get('testcase_technical_consistency', 0)
            print(f"    {i}. {model}: {score:.4f}")
    
    print(f"\n{'='*130}")
    
    # 평가 지표 해석 가이드
    print(f"\n📝 평가 지표 해석 가이드:")
    print(f"  🎯 Functional + Multi + Tech = 3가지 독립적 성능 지표")
    print(f"  📊 각 지표 0~1 범위, 높을수록 해당 측면에서 우수")
    print(f"  🔍 Functional: 같은 기능 테스트들의 임베딩 유사성")
    print(f"  🌏 Multi: 한국어-영어 키워드 쌍의 의미적 매칭")
    print(f"  🔧 Tech: 동일 기술 용어들의 임베딩 일관성")
    print(f"  💡 용도에 따라 중요한 지표를 선택하여 모델 결정")

def load_embedding_list_from_json(file_path):
    """
    JSON 파일에서 embedding_list를 로드합니다.
    
    Args:
        file_path: JSON 파일 경로
    
    Returns:
        embedding_list: 이슈별 텍스트 리스트들의 리스트
    """
    import json
    import os
    
    try:
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            print("📁 현재 디렉토리의 파일들:")
            for f in os.listdir('.'):
                if f.endswith('.json'):
                    print(f"  - {f}")
            return None
        
        print(f"📂 JSON 파일 로딩 중: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 다양한 JSON 구조에 대응
        embedding_list = None
        
        # 케이스 1: 직접 embedding_list가 있는 경우
        if 'embedding_list' in data:
            embedding_list = data['embedding_list']
            print(f"✅ 'embedding_list' 키에서 데이터 로드")
        
        # 케이스 2: 리스트 형태로 되어 있는 경우
        elif isinstance(data, list):
            embedding_list = data
            print(f"✅ 루트 레벨에서 리스트 데이터 로드")
        
        # 케이스 3: 이슈별로 키가 있는 경우 (예: {"issue_0": [...], "issue_1": [...]})
        elif isinstance(data, dict):
            # 이슈 키들을 찾아서 정렬
            issue_keys = [k for k in data.keys() if k.startswith('issue_') or 'COMMONR-' in k]
            if issue_keys:
                # 숫자 순으로 정렬
                if 'COMMONR-' in issue_keys[0]:
                    issue_keys.sort(key=lambda x: int(x.split('-')[1]) if '-' in x and x.split('-')[1].isdigit() else 0)
                else:
                    issue_keys.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
                
                embedding_list = [data[key] for key in issue_keys]
                print(f"✅ {len(issue_keys)}개 이슈 키에서 데이터 로드")
            else:
                # 모든 값들을 리스트로 변환
                embedding_list = list(data.values())
                print(f"✅ 딕셔너리 값들을 리스트로 변환")
        
        if embedding_list is None:
            print(f"❌ 지원되지 않는 JSON 구조입니다.")
            print(f"📋 데이터 구조: {type(data)}")
            if isinstance(data, dict):
                print(f"📋 키들: {list(data.keys())[:5]}...")
            return None
        
        # 데이터 검증
        if not isinstance(embedding_list, list):
            print(f"❌ embedding_list가 리스트가 아닙니다: {type(embedding_list)}")
            return None
        
        # 각 이슈의 텍스트들 확인
        valid_issues = 0
        total_texts = 0
        
        for i, issue_texts in enumerate(embedding_list):
            if isinstance(issue_texts, list):
                text_count = sum(1 for text in issue_texts if isinstance(text, str) and text.strip())
                if text_count > 0:
                    valid_issues += 1
                    total_texts += text_count
        
        print(f"📊 로드 완료:")
        print(f"  - 총 이슈 수: {len(embedding_list)}")
        print(f"  - 유효한 이슈 수: {valid_issues}")
        print(f"  - 총 텍스트 수: {total_texts}")
        
        if valid_issues == 0:
            print(f"❌ 유효한 텍스트가 있는 이슈가 없습니다.")
            return None
        
        return embedding_list
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"❌ 파일 로드 오류: {e}")
        return None

def main():
    """메인 실행 함수 - JSON에서 데이터 로드"""
    
    # JSON 파일에서 embedding_list 로드
    print("🔧 모델 평가 프로그램 (TestCase 특화)")
    print("=" * 50)
    file_path = "evaluate_json/test_file.json"    
    embedding_list = load_embedding_list_from_json(file_path)
    
    if embedding_list is None:
        print("\n⚠️ JSON 로드 실패, 테스트 데이터 사용")
        # 폴백: 작은 테스트 데이터
        embedding_list = [
            ["Login test step 1", "Login test step 2"],
            ["Dashboard test step 1", "Dashboard test step 2"],
            ["User management test step 1", "User management test step 2"]
        ]
        print(f"📋 테스트 데이터: {len(embedding_list)}개 이슈")
    
    # 한국어-영어 혼합 환경에 최적화된 다국어 모델들
    target_models = [
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 다국어 작은 모델
        'sentence-transformers/all-MiniLM-L6-v2',  # 영어 기준선
        'BAAI/bge-m3',
        'intfloat/multilingual-e5-large-instruct',                         
        'jinaai/jina-embeddings-v3',               
        'Salesforce/SFR-Embedding-Mistral'
    ]
    
    # 데이터 크기에 따른 처리 전략 결정
    total_texts = sum(len(issue_texts) for issue_texts in embedding_list if isinstance(issue_texts, list))
    
    if total_texts > 500:
        print(f"⚠️ 대용량 데이터 감지 ({total_texts}개 텍스트)")
        print("🔄 텍스트 수를 제한하여 처리합니다.")
        max_texts = 200
    elif total_texts > 100:
        print(f"📊 중간 크기 데이터 ({total_texts}개 텍스트)")
        max_texts = total_texts
    else:
        print(f"📋 소규모 데이터 ({total_texts}개 텍스트)")
        max_texts = total_texts
    
    # 목표 모델들 테스트 (한 번에 하나씩)
    print(f"\n🎯 모델 개별 테스트 (최대 {max_texts}개 텍스트)")
    target_results = {}
    
    for model in target_models:
        print(f"\n🎯 {model} 개별 테스트")
        result = evaluate_models_safely(embedding_list, [model], max_texts_per_model=max_texts)
        target_results.update(result)
        
        # 실패하면 다음 모델로
        if result.get(model) is None:
            print(f"⚠️ {model} 실패, 다음 모델로 진행")
            continue
    
    # 전체 결과 출력
    print("\n🏁 최종 결과")
    print_results_table(target_results)

if __name__ == "__main__":
    main()