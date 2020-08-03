#pragma once
template <typename T>
class vec2
{
public:
	T x, y;
	vec2()
	{
		x = 0; y = 0;
	}
	vec2(T _x, T _y)
	{
		x = _x;
		y = _y;
	}
};

template<typename T>
vec2<T> operator+(const vec2<T> & left, const vec2<T> & right)
{
	return vec2<T>(left.x + right.x, left.y + right.y);
}

template<typename T>
vec2<T> operator-(const vec2<T> & left, const vec2<T> & right)
{
	return vec2<T>(left.x - right.x, left.y - right.y);
}

template<typename T>
vec2<T> operator*(const vec2<T> & left, T num)
{
	return vec2<T>(left.x * num, left.y * num);
}

template<typename T>
vec2<T> operator/(const vec2<T> & left, T num)
{
	return vec2<T>(left.x / num, left.y / num);
}

template<typename T>
bool operator==(const vec2<T> & left, T right)
{
	return left.x == right;
}

template<typename T>
bool operator!=(const vec2<T> & left, T right)
{
	return left.x != right;
}

template<typename T>
vec2<T> vecLerp(const vec2<T> & from, const vec2<T> & to, float alpha)
{
	return vec2<T>(from.x * (1 - alpha) + to.x * alpha, 
				   from.y * (1 - alpha) + to.y * alpha);
}