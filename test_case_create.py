import requests
import os
import json
import shutil

# --- 1. 사용자 설정 ---
JIRA_URL = "https://jira.suprema.co.kr"
JIRA_USERNAME = "dhwoo"
JIRA_PASSWORD = "android0237@"

PROJECT_KEY = "COMMONR"
ISSUE_TYPE_NAME = "Test"

JQL_QUERY = f'project = "{PROJECT_KEY}" AND issuetype = "{ISSUE_TYPE_NAME}"'
OUTPUT_FOLDER_NAME = "jira_test_cases_txt"
JIRA_AUTH = (JIRA_USERNAME, JIRA_PASSWORD)


# --- 2. Jira에서 모든 이슈 검색 (페이지네이션 적용) ---
def get_all_jira_test_cases(jql):
    """JQL로 모든 이슈를 검색합니다."""
    all_issues = []
    start_at = 0
    max_results = 100

    while True:
        url = f"{JIRA_URL}/rest/api/2/search"
        headers = {"Accept": "application/json"}
        params = {'jql': jql, 'fields': 'summary,description', 'startAt': start_at, 'maxResults': max_results}

        try:
            response = requests.get(url, headers=headers, params=params, auth=JIRA_AUTH)
            response.raise_for_status()
            data = response.json()
            issues_on_page = data.get('issues', [])
            if not issues_on_page:
                break
            all_issues.extend(issues_on_page)
            start_at += len(issues_on_page)
            print(f"현재까지 {len(all_issues)} / {data['total']} 개 이슈를 가져왔습니다...")
        except requests.exceptions.RequestException as e:
            print(f"Jira 이슈 검색 오류: {e}")
            break
    return all_issues

# --- 3. Xray 상세 스텝 정보 가져오기 ---
def get_xray_test_steps(issue_key):
    """주어진 테스트 케이스의 상세 스텝 정보를 Xray API로 가져옵니다."""
    url = f"{JIRA_URL}/rest/raven/latest/api/test/{issue_key}/step"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, auth=JIRA_AUTH)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return []

# --- 4. 각 이슈와 상세 스텝을 파일로 저장 ---
def save_issues_to_files(issues):
    """가져온 이슈와 상세 스텝을 각기 다른 텍스트 파일로 저장합니다."""
    if os.path.exists(OUTPUT_FOLDER_NAME):
        print(f"\n기존 폴더 '{OUTPUT_FOLDER_NAME}'를 삭제합니다.")
        shutil.rmtree(OUTPUT_FOLDER_NAME)
    
    os.makedirs(OUTPUT_FOLDER_NAME)
    print(f"결과를 저장할 새 폴더 '{OUTPUT_FOLDER_NAME}'를 생성합니다.")

    for issue in issues:
        issue_key = issue['key']
        summary = issue['fields'].get('summary', '제목 없음')
        # Description 필드 가져오기
        description = issue['fields'].get('description', '설명 없음')
        
        steps = get_xray_test_steps(issue_key)
        
        file_name = f"{issue_key}.txt"
        file_path = os.path.join(OUTPUT_FOLDER_NAME, file_name)
        file_content = f"■ 이슈 키: {issue_key}\n"
        file_content += f"■ 제목: {summary}\n"
        file_content += f"■ 설명: {description}\n"
        file_content += "=" * 60 + "\n\n"
        
        if not steps:
            file_content += ">> 등록된 테스트 스텝이 없습니다."
        else:
            file_content += "■ 테스트 스텝\n\n"
            for index, step in enumerate(steps):
                step_obj = step.get('step')
                action = step_obj.get('raw', '').strip() if isinstance(step_obj, dict) else 'N/A'

                data_obj = step.get('data')
                data = data_obj.get('raw', '').strip() if isinstance(data_obj, dict) else 'N/A'
                
                result_obj = step.get('result')
                result = result_obj.get('raw', '').strip() if isinstance(result_obj, dict) else 'N/A'
                
                if not action: action = 'N/A'
                if not data: data = 'N/A'
                if not result: result = 'N/A'
                
                # ✅ 이 부분을 요청에 맞게 수정했습니다.
                file_content += f"[Step {index + 1}]\n"
                file_content += "스텝 (Step)\n"
                file_content += f"{action}\n"
                file_content += "\n데이터 (Data)\n"
                file_content += f"{data}\n"
                file_content += "\n예상 결과 (Expected Result)\n"
                file_content += f"{result}\n"
                file_content += "------------------------------------------------------------\n"
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_content)

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print(f"✅ [ {PROJECT_KEY} ] 프로젝트의 '{ISSUE_TYPE_NAME}' 이슈를 대상으로 검색을 시작합니다.")
    print(f"JQL 쿼리: {JQL_QUERY}")
    print("-" * 50)

    print("1. Jira에서 모든 테스트 케이스 이슈를 가져옵니다.")
    all_jira_issues = get_all_jira_test_cases(JQL_QUERY)
    
    if all_jira_issues:
        print(f"\n총 {len(all_jira_issues)}개의 테스트 케이스를 모두 찾았습니다.")
        print("2. 각 이슈의 상세 스텝을 포함하여 개별 파일로 저장합니다.")
        save_issues_to_files(all_jira_issues)
        print(f"\n✅ 완료! '{OUTPUT_FOLDER_NAME}' 폴더에 모든 파일이 저장되었습니다.")
    else:
        print("\nJQL에 해당하는 테스트 케이스를 찾지 못했습니다.")