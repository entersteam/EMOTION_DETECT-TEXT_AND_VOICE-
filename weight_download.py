import gdown

# 파일의 Google 드라이브 공유 가능한 링크에서 'id=' 뒷부분을 복사하여 사용합니다.
# 예를 들어, https://drive.google.com/file/d/abcdefg12345/view 의 경우 'abcdefg12345'를 사용합니다.

file_id = '1VpjCn6l4o_NMzhpLh-03D6nOzcWS9zg_'
output_file = './best.pt'  # 저장할 로컬 파일명 및 확장자

gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)