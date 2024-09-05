import axios from 'axios';

// 创建一个 axios 实例
const service = axios.create({
  baseURL: 'http://localhost:5173', // 你的 API 基础路径
  timeout: 5000, // 请求超时时间
});

// 请求拦截器
service.interceptors.request.use(
  config => {
    // 在发送请求之前做些什么，例如添加 token
    // config.headers['Authorization'] = `Bearer ${store.state.token}`;
    return config;
  },
  error => {
    // 处理请求错误
    return Promise.reject(error);
  }
);

// 响应拦截器
service.interceptors.response.use(
  response => {
    // 对响应数据做点什么
    const res = response.data;
    // console.log(response)
    if (response.status !== 200) {
      console.log("response.status !== 200")
      // 假设返回数据格式中有一个 code 字段表示错误或成功
      // 你可以根据 res.code 做一些处理
      return Promise.reject(new Error(res.message || 'Error'));
    } else {
      return res;
    }
  },
  error => {
    console.log("error process")
    // 处理响应错误
    return Promise.reject(error);
  }
);

// 封装 GET 请求
export const get = (url, params = {}) => {
  return service.get(url, { params });
};

// 封装 POST 请求
export const post = (url, data = {}) => {
  return service.post(url, data);
};

// 封装 PUT 请求
export const put = (url, data = {}) => {
  return service.put(url, data);
};

// 封装 DELETE 请求
export const del = (url, params = {}) => {
  return service.delete(url, { params });
};

// 你可以根据需要封装更多的方法，例如 PATCH
