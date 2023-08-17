// placeholder url if imgUrl is empty
export function placeholder(imgUrl) {
  if (imgUrl) {
    return imgUrl;
  }
  return 'https://via.placeholder.com/300x300?text=Loading';
}