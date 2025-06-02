import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import zipfile
from io import BytesIO

def detect_circle(image_path, save_result=True, output_path='detected_circles.jpg'):
    """
    检测图像中的圆形并返回圆的参数
    """
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"错误：无法读取图像 {image_path}")
        return None
      
    original = img.copy()
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    GrayImage = cv2.medianBlur(GrayImage, 5)
    ret, th1 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
    th1_inv = cv2.bitwise_not(th1)
    contours, _ = cv2.findContours(th1_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    if not contours:
        st.warning("未检测到圆")
        return None
      
    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)
  
    # 绘制结果
    cv2.circle(original, center, radius, (255, 0, 0), 10)
    cv2.circle(original, center, 2, (0, 0, 255), 100)
  
    if save_result:
        cv2.imwrite(output_path, original)
  
    return (center[0], center[1], radius), original

def crop_circle(image_path, padding_ratio=0.1, save_cropped=True, output_dir='cropped_circles'):
    """
    裁剪图像中的圆形区域
    """
    # 检测圆形
    result = detect_circle(image_path, save_result=False)
    if result is None:
        return None, None, None
  
    circle_params, detected_img = result
    center_x, center_y, radius = circle_params
  
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
  
    # 计算裁剪区域
    padding = int(radius * padding_ratio)
    crop_radius = radius + padding
  
    # 确保裁剪区域在图像范围内
    x1 = max(0, center_x - crop_radius)
    y1 = max(0, center_y - crop_radius)
    x2 = min(width, center_x + crop_radius)
    y2 = min(height, center_y + crop_radius)
  
    # 裁剪图像
    cropped_img = img[y1:y2, x1:x2]
  
    # 调整圆心坐标到新的坐标系
    new_center_x = center_x - x1
    new_center_y = center_y - y1
  
    # 创建圆形遮罩
    crop_height, crop_width = cropped_img.shape[:2]
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cv2.circle(mask, (new_center_x, new_center_y), radius, 255, -1)
  
    # 只保留圆形区域
    result_img = cropped_img.copy()
    result_img[mask == 0] = 0  # 圆外区域设为黑色
  
    if save_cropped:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
        cv2.imwrite(cropped_path, result_img)
  
    return result_img, radius, detected_img

def normalize_circle(image, current_radius, target_radius=1350):
    """
    将圆形图像标准化为指定半径
    """
    if image is None:
        return None
      
    # 计算缩放比例
    scale_factor = target_radius / current_radius
  
    # 计算新的尺寸
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
  
    # 缩放图像
    normalized_img = cv2.resize(image, (new_width, new_height), 
                              interpolation=cv2.INTER_LANCZOS4)
  
    return normalized_img

def process_single_image(uploaded_file, padding_ratio, target_radius):
    """
    处理单个图像
    """
    # 保存上传的文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
  
    try:
        # 裁剪圆形
        cropped_img, radius, detected_img = crop_circle(tmp_path, padding_ratio, save_cropped=False)
      
        if cropped_img is not None:
            # 标准化
            normalized_img = normalize_circle(cropped_img, radius, target_radius)
          
            return {
                'detected': detected_img,
                'cropped': cropped_img,
                'normalized': normalized_img,
                'radius': radius
            }
        else:
            return None
    finally:
        # 清理临时文件
        os.unlink(tmp_path)

def create_download_zip(processed_images, image_names):
    """
    创建包含所有处理结果的ZIP文件
    """
    zip_buffer = BytesIO()
  
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, (result, name) in enumerate(zip(processed_images, image_names)):
            if result is not None:
                base_name = os.path.splitext(name)[0]
              
                # 保存检测结果
                _, detected_encoded = cv2.imencode('.jpg', result['detected'])
                zip_file.writestr(f"{base_name}_detected.jpg", detected_encoded.tobytes())
              
                # 保存裁剪结果
                _, cropped_encoded = cv2.imencode('.jpg', result['cropped'])
                zip_file.writestr(f"{base_name}_cropped.jpg", cropped_encoded.tobytes())
              
                # 保存标准化结果
                _, normalized_encoded = cv2.imencode('.jpg', result['normalized'])
                zip_file.writestr(f"{base_name}_normalized.jpg", normalized_encoded.tobytes())
  
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.set_page_config(
        page_title="麦框识别工具",
        page_icon="🔍",
        layout="wide"
    )
  
    st.title("🔍 麦框识别工具")
    st.markdown("---")
  
    # 侧边栏参数设置
    st.sidebar.header("⚙️ 参数设置")
  
    padding_ratio = st.sidebar.slider(
        "裁剪边距比例",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="相对于半径的边距比例，0表示紧贴圆边缘"
    )
  
    target_radius = st.sidebar.number_input(
        "目标半径（像素）",
        min_value=100,
        max_value=3000,
        value=1350,
        step=50,
        help="标准化后圆形的半径"
    )
  
    # 主界面
    tab1, tab2 = st.tabs(["📷 单图处理", "📁 批量处理"])
  
    with tab1:
        st.header("单图处理模式")
      
        uploaded_file = st.file_uploader(
            "选择图像文件",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="支持 JPG, PNG, BMP 格式"
        )
      
        if uploaded_file is not None:
            # 显示原始图像
            col1, col2 = st.columns(2)
          
            with col1:
                st.subheader("📋 原始图像")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)  # Updated from use_column_width
                st.info(f"图像尺寸: {image.size[0]} × {image.size[1]}")
          
            # 处理按钮
            if st.button("🚀 开始处理", type="primary"):
                with st.spinner("正在处理图像..."):
                    result = process_single_image(uploaded_file, padding_ratio, target_radius)
              
                if result is not None:
                    with col2:
                        st.subheader("✅ 处理结果")
                      
                        # 显示检测结果
                        # st.write("**圆形检测结果:**")
                        detected_rgb = cv2.cvtColor(result['detected'], cv2.COLOR_BGR2RGB)
                        st.image(detected_rgb, use_container_width=True)  # Updated from use_column_width
                        st.success(f"检测到圆形，半径: {result['radius']} 像素")
                  
                    # 显示处理步骤
                    st.subheader("📊 处理步骤展示")
                  
                    step_col1, step_col2, step_col3 = st.columns(3)
                  
                    with step_col1:
                        st.write("**1️⃣ 圆形检测**")
                        detected_rgb = cv2.cvtColor(result['detected'], cv2.COLOR_BGR2RGB)
                        st.image(detected_rgb, use_container_width=True)  # Updated from use_column_width
                  
                    with step_col2:
                        st.write("**2️⃣ 圆形裁剪**")
                        cropped_rgb = cv2.cvtColor(result['cropped'], cv2.COLOR_BGR2RGB)
                        st.image(cropped_rgb, use_container_width=True)  # Updated from use_column_width
                  
                    with step_col3:
                        st.write("**3️⃣ 尺寸标准化**")
                        normalized_rgb = cv2.cvtColor(result['normalized'], cv2.COLOR_BGR2RGB)
                        st.image(normalized_rgb, use_container_width=True)  # Updated from use_column_width
                  
                    # 下载按钮
                    st.subheader("💾 下载结果")
                  
                    download_col1, download_col2, download_col3 = st.columns(3)
                  
                    with download_col1:
                        _, detected_encoded = cv2.imencode('.jpg', result['detected'])
                        st.download_button(
                            label="下载检测结果",
                            data=detected_encoded.tobytes(),
                            file_name=f"{uploaded_file.name.split('.')[0]}_detected.jpg",
                            mime="image/jpeg"
                        )
                  
                    with download_col2:
                        _, cropped_encoded = cv2.imencode('.jpg', result['cropped'])
                        st.download_button(
                            label="下载裁剪结果",
                            data=cropped_encoded.tobytes(),
                            file_name=f"{uploaded_file.name.split('.')[0]}_cropped.jpg",
                            mime="image/jpeg"
                        )
                  
                    with download_col3:
                        _, normalized_encoded = cv2.imencode('.jpg', result['normalized'])
                        st.download_button(
                            label="下载标准化结果",
                            data=normalized_encoded.tobytes(),
                            file_name=f"{uploaded_file.name.split('.')[0]}_normalized.jpg",
                            mime="image/jpeg"
                        )
              
                else:
                    st.error("处理失败！请检查图像是否包含可检测的圆形。")
  
    with tab2:
        st.header("批量处理模式")
      
        uploaded_files = st.file_uploader(
            "选择多个图像文件",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="可以同时选择多个图像文件进行批量处理"
        )
      
        if uploaded_files:
            st.success(f"已选择 {len(uploaded_files)} 个文件")
          
            # 显示文件列表
            with st.expander("📋 查看文件列表"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name}")
          
            if st.button("🚀 开始批量处理", type="primary"):
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
              
                processed_images = []
                image_names = []
              
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"正在处理: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                  
                    result = process_single_image(uploaded_file, padding_ratio, target_radius)
                    processed_images.append(result)
                    image_names.append(uploaded_file.name)
                  
                    # 更新进度条
                    progress_bar.progress((i + 1) / len(uploaded_files))
              
                status_text.text("处理完成！")
              
                # 统计结果
                success_count = sum(1 for result in processed_images if result is not None)
                failure_count = len(processed_images) - success_count
              
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总文件数", len(uploaded_files))
                with col2:
                    st.metric("成功处理", success_count)
                with col3:
                    st.metric("处理失败", failure_count)
              
                # 显示处理结果预览
                if success_count > 0:
                    st.subheader("📊 处理结果预览")
                  
                    # 显示前几个成功处理的结果
                    preview_count = min(3, success_count)
                    cols = st.columns(preview_count)
                  
                    preview_index = 0
                    for i, (result, name) in enumerate(zip(processed_images, image_names)):
                        if result is not None and preview_index < preview_count:
                            with cols[preview_index]:
                                st.write(f"**{name}**")
                                normalized_rgb = cv2.cvtColor(result['normalized'], cv2.COLOR_BGR2RGB)
                                st.image(normalized_rgb, use_container_width=True)  # Updated from use_column_width
                                st.caption(f"半径: {result['radius']}px")
                            preview_index += 1
                    
                    # 创建下载ZIP文件
                    st.subheader("💾 下载批量处理结果")
                    
                    with st.spinner("正在打包结果文件..."):
                        zip_buffer = create_download_zip(processed_images, image_names)
                    
                    st.download_button(
                        label="📦 下载所有结果 (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="batch_processed_circles.zip",
                        mime="application/zip",
                        help="包含所有成功处理的图像的检测、裁剪和标准化结果"
                    )
                    
                    # 详细结果表格
                    with st.expander("📋 详细处理结果"):
                        result_data = []
                        for i, (result, name) in enumerate(zip(processed_images, image_names)):
                            if result is not None:
                                result_data.append({
                                    "文件名": name,
                                    "状态": "✅ 成功",
                                    "检测半径": f"{result['radius']}px",
                                    "标准化后尺寸": f"{result['normalized'].shape[1]}×{result['normalized'].shape[0]}"
                                })
                            else:
                                result_data.append({
                                    "文件名": name,
                                    "状态": "❌ 失败",
                                    "检测半径": "N/A",
                                    "标准化后尺寸": "N/A"
                                })
                        
                        st.dataframe(result_data, use_container_width=True)
                
                if failure_count > 0:
                    st.warning(f"有 {failure_count} 个文件处理失败，可能原因：图像中没有可检测的圆形区域")
    # 底部信息
    st.markdown("---")
    with st.expander("ℹ️ 使用说明"):
        st.markdown("""
        ### 功能说明
        
        **麦框识别工具** 可以自动检测图像中的麦框区域，并进行裁剪和标准化处理。
        
        ### 处理流程
        1. **圆形检测**: 使用轮廓检测算法找到图像中最大的圆形区域
        2. **圆形裁剪**: 根据检测到的圆心和半径，裁剪出圆形区域
        3. **尺寸标准化**: 将圆形缩放到指定的标准半径
        
        ### 参数说明
        - **裁剪边距比例**: 裁剪时在圆形边缘外保留的额外空间，相对于半径的比例
        - **目标半径**: 标准化后圆形的半径大小（以像素为单位）
        
        ### 注意事项
        - 图像中应包含明显的圆形轮廓
        - 建议使用对比度较高的图像以获得更好的检测效果
        - 支持的格式：JPG、PNG、BMP
        """)
    
    with st.expander("🔧 技术参数"):
        st.markdown("""
        ### 算法参数
        - **图像预处理**: 中值滤波 (kernel=5)
        - **二值化阈值**: 127
        - **轮廓检测**: RETR_EXTERNAL + CHAIN_APPROX_SIMPLE
        - **缩放插值**: INTER_LANCZOS4 (高质量插值)
        
        ### 性能指标
        - **检测精度**: 基于最大轮廓的最小外接圆
        - **处理速度**: 单张图像 < 1秒 (取决于图像大小)
        - **支持尺寸**: 最大 10MB 图像文件
        """)
if __name__ == "__main__":
    main()