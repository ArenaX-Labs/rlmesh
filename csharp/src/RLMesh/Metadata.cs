using System;
using System.Collections;
using System.Collections.Generic;

namespace RLMesh;

/// <summary>Experimental. A mutable insertion-ordered map. Do not mutate a map used as a hash key.</summary>
public sealed class Metadata : IDictionary<string, Value>, IReadOnlyDictionary<string, Value>, IEquatable<Metadata>
{
    private readonly Dictionary<string, Value> values = new(StringComparer.Ordinal);
    private readonly List<string> order = new();
    public Metadata() { }
    public Metadata(IEnumerable<KeyValuePair<string, Value>> pairs)
    {
        if (pairs == null) throw new ArgumentNullException(nameof(pairs));
        foreach (var pair in pairs) this[pair.Key] = pair.Value;
    }
    public Value this[string key]
    {
        get => values[key];
        set
        {
            Validate(key, value);
            if (!values.ContainsKey(key)) order.Add(key);
            values[key] = value;
        }
    }
    private static void Validate(string key, Value value)
    {
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value is null) throw new ArgumentNullException(nameof(value));
        Interop.Utf8.Encode(key);
    }
    public bool TryGetValue(string key, out Value value) => values.TryGetValue(key, out value!);
    public bool ContainsKey(string key) => values.ContainsKey(key);
    public void Add(string key, Value value)
    {
        Validate(key, value);
        values.Add(key, value);
        order.Add(key);
    }
    public bool Remove(string key)
    {
        if (!values.Remove(key)) return false;
        order.Remove(key);
        return true;
    }
    public int Count => values.Count;
    public ICollection<string> Keys => order.AsReadOnly();
    public ICollection<Value> Values
    {
        get
        {
            var result = new List<Value>(Count);
            foreach (string key in order) result.Add(values[key]);
            return result.AsReadOnly();
        }
    }
    public IEnumerator<KeyValuePair<string, Value>> GetEnumerator()
    {
        foreach (string key in order) yield return new(key, values[key]);
    }
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    IEnumerable<string> IReadOnlyDictionary<string, Value>.Keys => Keys;
    IEnumerable<Value> IReadOnlyDictionary<string, Value>.Values => Values;
    bool ICollection<KeyValuePair<string, Value>>.IsReadOnly => false;
    void ICollection<KeyValuePair<string, Value>>.Add(KeyValuePair<string, Value> item) => Add(item.Key, item.Value);
    void ICollection<KeyValuePair<string, Value>>.Clear() { values.Clear(); order.Clear(); }
    bool ICollection<KeyValuePair<string, Value>>.Contains(KeyValuePair<string, Value> item) => values.TryGetValue(item.Key, out var value) && value == item.Value;
    void ICollection<KeyValuePair<string, Value>>.CopyTo(KeyValuePair<string, Value>[] array, int arrayIndex)
    {
        if (array == null) throw new ArgumentNullException(nameof(array));
        if (arrayIndex < 0 || arrayIndex > array.Length) throw new ArgumentOutOfRangeException(nameof(arrayIndex));
        if (array.Length - arrayIndex < Count) throw new ArgumentException("array is too small");
        foreach (var item in this) array[arrayIndex++] = item;
    }
    bool ICollection<KeyValuePair<string, Value>>.Remove(KeyValuePair<string, Value> item) => ((ICollection<KeyValuePair<string, Value>>)this).Contains(item) && Remove(item.Key);
    public bool Equals(Metadata? other)
    {
        if (other is null || Count != other.Count) return false;
        foreach (var pair in this)
            if (!other.TryGetValue(pair.Key, out var value) || pair.Value != value) return false;
        return true;
    }
    public override bool Equals(object? obj) => obj is Metadata other && Equals(other);
    public override int GetHashCode() => Count;
    public static bool operator ==(Metadata? left, Metadata? right) => left is null ? right is null : left.Equals(right);
    public static bool operator !=(Metadata? left, Metadata? right) => !(left == right);
}
