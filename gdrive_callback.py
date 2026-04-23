import os
import logging
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# 设置日志，以便在训练输出中看到上传信息
log = logging.getLogger(__name__)

class GoogleDriveUploader(Callback):
    """
    一个自定义的PyTorch Lightning回调，用于在新的最佳模型被保存后，
    自动将其上传到Google Drive，并可选择删除本地副本。
    """
    def __init__(self, drive_folder_id, credentials_file="credentials.json", delete_local_after_upload=False):
        """
        初始化回调。
        Args:
            drive_folder_id (str): 你的Google Drive目标文件夹的ID。
            credentials_file (str): 你的Google Cloud凭证文件路径 (credentials.json)。
            delete_local_after_upload (bool): 是否在成功上传后删除本地的ckpt文件。
                                              警告：这可以节省空间，但如果需要从该检查点恢复，可能会有问题。
        """
        super().__init__()
        self.drive_folder_id = drive_folder_id
        self.credentials_file = credentials_file
        self.delete_local_after_upload = delete_local_after_upload
        self.drive_service = None
        self.uploaded_paths = set()

    def setup(self, trainer, pl_module, stage: str):
        """
        在训练开始时被调用，用于执行一次性的设置，如API认证。
        """
        if stage == 'fit':
            log.info("正在进行Google Drive认证...")
            try:
                self.drive_service = self._authenticate()
                if self.drive_service:
                    log.info("Google Drive认证成功！")
                else:
                    log.error("Google Drive认证失败，上传功能将不可用。")
            except Exception as e:
                log.error(f"Google Drive认证过程中发生错误: {e}")


    def on_validation_epoch_end(self, trainer, pl_module):
        """
        在每个验证epoch结束后被调用。这是检查和上传新模型的最佳时机。
        """
        if not self.drive_service:
            return # 如果认证失败，则不执行任何操作

        # 从trainer的回调列表中找到ModelCheckpoint
        model_checkpoint_callback = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                model_checkpoint_callback = cb
                break
        
        if not model_checkpoint_callback:
            log.warning("未找到ModelCheckpoint回调，无法执行上传。")
            return

        # 遍历ModelCheckpoint跟踪的所有模型路径
        # 当save_top_k=-1时，这应该包含所有保存的检查点
        for path in list(model_checkpoint_callback.best_k_models.keys()):
            # 检查是否是新文件（未上传过）且文件确实存在
            if path not in self.uploaded_paths and os.path.exists(path):
                log.info(f"检测到新模型: {os.path.basename(path)}")
                
                # 上传这个新文件
                success = self._upload_single_file(path)

                if success:
                    # 更新记录，避免重复上传
                    self.uploaded_paths.add(path)
                    
                    if self.delete_local_after_upload:
                        log.info(f"成功上传后，正在删除本地文件: {path}")
                        try:
                            os.remove(path)
                            log.info(f"本地文件 '{os.path.basename(path)}' 已删除。")
                        except OSError as e:
                            log.error(f"删除本地文件时出错: {e}")

    def _authenticate(self):
        """
        执行Google API认证流程。
        """
        creds = None
        token_path = "/notebooks/token.json"
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, [ "https://www.googleapis.com/auth/drive" ])

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_file,
                        [ "https://www.googleapis.com/auth/drive" ],
                        redirect_uri='urn:ietf:wg:oauth:2.0:oob'
                    )
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    print('-' * 80)
                    print('请在你的本地浏览器中访问以下URL进行授权:')
                    print(auth_url)
                    print('-' * 80)
                    code = input('授权后，请将Google提供给你的授权码粘贴到这里并按Enter: ')
                    flow.fetch_token(code=code)
                    creds = flow.credentials
                except FileNotFoundError:
                    log.error(f"错误: 凭证文件 '{self.credentials_file}' 未找到。请确保它在正确的位置。")
                    return None

            with open(token_path, "w") as token:
                token.write(creds.to_json())
        
        try:
            return build("drive", "v3", credentials=creds)
        except HttpError as error:
            log.error(f"构建Google Drive服务时发生错误: {error}")
            return None

    def _upload_single_file(self, local_path):
        """
        上传单个文件到指定的Google Drive文件夹。
        """
        if not os.path.isfile(local_path):
            log.error(f"尝试上传失败，路径不是一个文件: {local_path}")
            return False
            
        filename = os.path.basename(local_path)
        log.info(f"正在上传 {filename} 到Google Drive...")
        
        try:
            file_metadata = {
                "name": filename,
                "parents": [self.drive_folder_id]
            }
            media = MediaFileUpload(local_path, resumable=True)
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()
            log.info(f"上传成功! 文件 '{filename}' 的ID是: {file.get('id')}")
            return True
        except HttpError as error:
            log.error(f"上传 {filename} 时发生错误: {error}")
            return False
